import clip
from tqdm import tqdm
import kaolin.ops.mesh
import kaolin as kal
import torch
from args import parse_args
from neural_style_field import NeuralStyleField
from utils import device
from render import Renderer
from mesh_scannet import Mesh
from utils import clip_model
from Normalization import MeshNormalizer
from utils import preprocess, add_vertices, sample_bary
import numpy as np
import random
import copy
import torchvision
import os
from PIL import Image
import argparse
from pathlib import Path
from torchvision import transforms
from IPython import embed
from convert import HSVLoss as HSV

# 1. camera position
# 2. H of HSV range
# 3. random camera pose until fore ground larger that 80%
# 4. avoid faces

def run_branched(args):
    dir = args.output_dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Constrain all sources of randomness
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    objbase, extension = os.path.splitext(os.path.basename(args.obj_path))

    render = Renderer()
    full_mesh = Mesh(args.obj_path)
    init_full_mesh = copy.deepcopy(full_mesh)

    #focus_one_thing = True
    full_pred_rgb = torch.zeros([full_mesh.vertices.shape[0], 3], dtype=torch.float32)
    full_pred_normal = torch.zeros([full_mesh.vertices.shape[0], 3], dtype=torch.float32)
    full_final_mask = torch.zeros([full_mesh.vertices.shape[0]], dtype=torch.float32)
    for label_order, label in enumerate(args.label):

        if args.focus_one_thing and not args.render_all_grad_one:
            init_mesh, _, _ = init_full_mesh.mask_mesh(label)
        else:
            init_mesh = copy.deepcopy(init_full_mesh)
        init_mesh_colors = torch.clone(kaolin.ops.mesh.index_vertices_by_faces(
            init_mesh.colors.unsqueeze(0),
            init_mesh.faces).squeeze())

        if args.focus_one_thing:
            if not (full_mesh.labels==label).any():
                print(f"label {label} is not in this mesh")
                continue
            ver_mask, face_mask = full_mesh.get_mask(label)
            if args.render_all_grad_one:
                mesh = copy.deepcopy(full_mesh)
                old_indice_to_new = None
                new_indice_to_old = None
            else:
                mesh, old_indice_to_new, new_indice_to_old = full_mesh.mask_mesh(label)
        else:
            mesh = copy.deepcopy(full_mesh)
            old_indice_to_new = None
            new_indice_to_old = None
            ver_mask = None
            face_mask = None

        MeshNormalizer(mesh)()

        # with_prior_color: start with original color
        if args.with_prior_color:
            prior_color = kaolin.ops.mesh.index_vertices_by_faces(
                        mesh.colors.unsqueeze(0),
                        mesh.faces).squeeze()
        else:
            prior_color = torch.full(size=(mesh.faces.shape[0], 3, 3), fill_value=0.5, device=device)

        background = None
        if args.background is not None:
            assert len(args.background) == 3
            background = torch.tensor(args.background).to(device)

        losses = []

        n_augs = args.n_augs
        # dir = args.output_dir
        clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        # CLIP Transform
        clip_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            clip_normalizer
        ])

        # Augmentation settings
        augment_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(1, 1)),
            transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
            clip_normalizer
        ])

        # Augmentations for normal network
        if args.cropforward :
            curcrop = args.normmincrop
        else:
            curcrop = args.normmaxcrop
        normaugment_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(curcrop, curcrop)),
            transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
            clip_normalizer
        ])
        cropiter = 0
        cropupdate = 0
        if args.normmincrop < args.normmaxcrop and args.cropsteps > 0:
            cropiter = round(args.n_iter / (args.cropsteps + 1))
            # cropupdate = (args.maxcrop - args.mincrop) / cropiter
            cropupdate = -0.9

            if not args.cropforward:
                cropupdate *= -1

        # Displacement-only augmentations
        displaugment_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(args.normmincrop, args.normmincrop)),
            transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
            clip_normalizer
        ])

        normweight = 1.0

        # MLP Settings
        input_dim = 6 if args.input_normals else 3
        if args.only_z:
            input_dim = 1
        mlp = NeuralStyleField(args.sigma, args.depth, args.width, 'gaussian', args.colordepth, args.normdepth,
                                    args.normratio, args.clamp, args.normclamp, niter=args.n_iter,
                                    progressive_encoding=args.pe, input_dim=input_dim, exclude=args.exclude).to(device)
        mlp.reset_weights()

        optim = torch.optim.Adam(mlp.parameters(), args.learning_rate, weight_decay=args.decay)
        activate_scheduler = args.lr_decay < 1 and args.decay_step > 0 and not args.lr_plateau
        if activate_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.decay_step, gamma=args.lr_decay)
        if not args.no_prompt:
            if args.prompt:
                prompt = ' '.join(args.prompt)
                prompt = prompt.split(',')[label_order]
                prompt_token = clip.tokenize([prompt]).to(device)
                encoded_text = clip_model.encode_text(prompt_token)

                # Save prompt
                with open(os.path.join(dir, prompt), "w") as f:
                    f.write("")

                # Same with normprompt
                norm_encoded = encoded_text
        if args.normprompt is not None:
            prompt = ' '.join(args.normprompt)
            prompt = prompt.split(',')[label_order]
            prompt_token = clip.tokenize([prompt]).to(device)
            norm_encoded = clip_model.encode_text(prompt_token)

            # Save prompt
            with open(os.path.join(dir, f"NORM {prompt}"), "w") as f:
                f.write("")

        if args.image:
            img = Image.open(args.image)
            img = preprocess(img).to(device)
            encoded_image = clip_model.encode_image(img.unsqueeze(0))
            if args.no_prompt:
                norm_encoded = encoded_image

        loss_check = None
        vertices = copy.deepcopy(mesh.vertices)
        network_input = copy.deepcopy(vertices)
        if args.symmetry == True:
            network_input[:,2] = torch.abs(network_input[:,2])

        if args.standardize == True:
            # Each channel into z-score
            network_input = (network_input - torch.mean(network_input, dim=0))/torch.std(network_input, dim=0)

        for i in tqdm(range(args.n_iter)):
            optim.zero_grad()

            sampled_mesh = mesh

            update_mesh(args, mlp, network_input, prior_color, sampled_mesh, vertices, ver_mask=ver_mask)
            loss = 0.0
            if args.with_hsv_loss:
                hsv_loss = 0.0
                h1, s1, v1 = HSV().get_hsv(sampled_mesh.face_attributes.permute(0,3,1,2))
                h2, s2, v2 = HSV().get_hsv(init_mesh_colors.unsqueeze(0).permute(0,3,1,2))
                hsv_loss += 0.9 * (h1 - h2).abs().mean()
                hsv_loss += 0.9 * (s1 - s2).abs().mean()
                hsv_loss += 0.9 * (v1 - v2).abs().mean()
                hsv_loss += (h1.mean() - h2.mean()).abs()
                hsv_loss += (s1.mean() - s2.mean()).abs()
                hsv_loss += (v1.mean() - v2.mean()).abs()
                hsv_loss += (h1.std() - h2.std()).abs()
                hsv_loss += (s1.std() - s2.std()).abs()
                hsv_loss += (v1.std() - v2.std()).abs()
                loss += hsv_loss.reshape(1) / 6


            rendered_images, elev, azim = render.render_center_out_views(sampled_mesh, num_views=args.n_views, lighting=args.lighting,
                                                                    show=args.show,
                                                                    center_azim=args.frontview_center[0],
                                                                    center_elev=args.frontview_center[1],
                                                                    std=args.frontview_std,
                                                                    return_views=True,
                                                                    background=background,
                                                                    rand_background=args.rand_background,
                                                                    rand_focal=args.rand_focal)

            if n_augs == 0:
                clip_image = clip_transform(rendered_images)
                encoded_renders = clip_model.encode_image(clip_image)
                if not args.no_prompt:
                    loss += torch.mean(torch.cosine_similarity(encoded_renders, encoded_text))

            # Check augmentation steps
            if args.cropsteps != 0 and cropupdate != 0 and i != 0 and i % args.cropsteps == 0:
                curcrop += cropupdate
                # print(curcrop)
                normaugment_transform = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(curcrop, curcrop)),
                    transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
                    clip_normalizer
                ])

            if n_augs > 0:
                # loss = 0.0
                for _ in range(n_augs):
                    augmented_image = augment_transform(rendered_images)
                    encoded_renders = clip_model.encode_image(augmented_image)
                    if not args.no_prompt:
                        if args.prompt:
                            if args.clipavg == "view":
                                if encoded_text.shape[0] > 1:
                                    loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                                    torch.mean(encoded_text, dim=0), dim=0)
                                else:
                                    loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True),
                                                                    encoded_text)
                                    # embed()
                            else:
                                loss -= torch.mean(torch.cosine_similarity(encoded_renders, encoded_text))
                    if args.image:
                        if encoded_image.shape[0] > 1:
                            loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                            torch.mean(encoded_image, dim=0), dim=0)
                        else:
                            loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True),
                                                            encoded_image)
                        # if args.image:
                        #     loss -= torch.mean(torch.cosine_similarity(encoded_renders,encoded_image))
            if args.splitnormloss:
                for param in mlp.mlp_normal.parameters():
                    param.requires_grad = False
            # loss.backward(retain_graph=True)

            # optim.step()

            # with torch.no_grad():
            #     losses.append(loss.item())

            # Normal augment transform
            # loss = 0.0
            if args.n_normaugs > 0:
                # loss = 0.0
                for _ in range(args.n_normaugs):
                    augmented_image = normaugment_transform(rendered_images)
                    encoded_renders = clip_model.encode_image(augmented_image)
                    if not args.no_prompt:
                        if args.prompt:
                            if args.clipavg == "view":
                                if norm_encoded.shape[0] > 1:
                                    loss -= normweight * torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                                                    torch.mean(norm_encoded, dim=0),
                                                                                    dim=0)
                                else:
                                    loss -= normweight * torch.cosine_similarity(
                                        torch.mean(encoded_renders, dim=0, keepdim=True),
                                        norm_encoded)
                            else:
                                loss -= normweight * torch.mean(
                                    torch.cosine_similarity(encoded_renders, norm_encoded))
                    if args.image:
                        if encoded_image.shape[0] > 1:
                            loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                            torch.mean(encoded_image, dim=0), dim=0)
                        else:
                            loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True),
                                                            encoded_image)
                        # if args.image:
                        #     loss -= torch.mean(torch.cosine_similarity(encoded_renders,encoded_image))
                if args.splitnormloss:
                    for param in mlp.mlp_normal.parameters():
                        param.requires_grad = True
                if args.splitcolorloss:
                    for param in mlp.mlp_rgb.parameters():
                        param.requires_grad = False
                if not args.no_prompt:
                    loss.backward(retain_graph=True)

            # Also run separate loss on the uncolored displacements
            if args.geoloss:
                default_color = torch.zeros(len(mesh.vertices), 3).to(device)
                default_color[:, :] = torch.tensor([0.5, 0.5, 0.5]).to(device)
                sampled_mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(default_color.unsqueeze(0),
                                                                                    sampled_mesh.faces)
                geo_renders, elev, azim = render.render_center_out_views(sampled_mesh, num_views=args.n_views,
                                                                    show=args.show,
                                                                    center_azim=args.frontview_center[0],
                                                                    center_elev=args.frontview_center[1],
                                                                    std=args.frontview_std,
                                                                    return_views=True,
                                                                    background=background)
                if args.n_normaugs > 0:
                    normloss = 0.0
                    ### avgview != aug
                    for _ in range(args.n_normaugs):
                        augmented_image = displaugment_transform(geo_renders)
                        encoded_renders = clip_model.encode_image(augmented_image)
                        if norm_encoded.shape[0] > 1:
                            normloss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                                torch.mean(norm_encoded, dim=0), dim=0)
                        else:
                            normloss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True),
                                                                norm_encoded)
                        if args.image:
                            if encoded_image.shape[0] > 1:
                                loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                                torch.mean(encoded_image, dim=0), dim=0)
                            else:
                                loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True),
                                                                encoded_image)  # if args.image:
                            #     loss -= torch.mean(torch.cosine_similarity(encoded_renders,encoded_image))
                    # if not args.no_prompt:
                    normloss.backward(retain_graph=True)
            optim.step()

            for param in mlp.mlp_normal.parameters():
                param.requires_grad = True
            for param in mlp.mlp_rgb.parameters():
                param.requires_grad = True

            if activate_scheduler:
                lr_scheduler.step()

            with torch.no_grad():
                losses.append(loss.item())

            # Adjust normweight if set
            if args.decayfreq is not None:
                if i % args.decayfreq == 0:
                    normweight *= args.cropdecay

            if i % 10 == 0:
                report_process(args, dir, i, loss, loss_check, losses, rendered_images, label)

        if args.focus_one_thing:
            pred_rgb, pred_normal = export_full_results(args, dir, losses, mesh, full_mesh, mlp, network_input, vertices,
                                old_indice_to_new=old_indice_to_new, new_indice_to_old=new_indice_to_old, label = label)
            if args.render_all_grad_one:
                cpu_ver_mask = ver_mask.detach().cpu()
                pred_normal = pred_rgb * cpu_ver_mask.unsqueeze(dim=-1)
                pred_rgb = pred_rgb * cpu_ver_mask.unsqueeze(dim=-1)
            full_pred_normal = full_pred_normal + pred_normal
            full_pred_rgb = full_pred_rgb + pred_rgb
            
            if args.render_all_grad_one:
                full_final_mask = full_final_mask + cpu_ver_mask
                del pred_normal
                del pred_rgb
                del cpu_ver_mask

        else:
            export_final_results(args, dir, losses, mesh, mlp, network_input, vertices)
    if args.with_prior_color:
        # if there's label, use 0.5 value, else use original color
        full_base_color = full_mesh.colors.detach().cpu()
        full_final_color = torch.clamp(full_pred_rgb + full_base_color, 0, 1)
    else:
        full_base_color = torch.full(size=(full_mesh.vertices.shape[0], 3), fill_value=0.5)
        pred_final_color = torch.clamp(full_pred_rgb + full_base_color, 0, 1)
        full_prior_color = full_mesh.colors.detach().cpu()
        full_final_color = pred_final_color*(full_final_mask.unsqueeze(dim=-1)) + full_prior_color*((1-full_final_mask).unsqueeze(dim=-1))

    # FixMe: input vertices should be fixed
    if args.color_only:
        full_mesh.vertices = full_mesh.vertices.detach().cpu()
    else:
        full_mesh.vertices = full_mesh.vertices.detach().cpu() + full_mesh.vertex_normals.detach().cpu() * full_pred_normal

    objbase, extension = os.path.splitext(os.path.basename(args.obj_path))
    full_mesh.export(os.path.join(dir, f"all_{objbase}_full_final.obj"), color=full_final_color)



def report_process(args, dir, i, loss, loss_check, losses, rendered_images, label):
    print('iter: {} loss: {}'.format(i, loss.item()))
    torchvision.utils.save_image(rendered_images, os.path.join(dir, 'label_{}_iter_{}.jpg'.format(label, i)))
    if args.lr_plateau and loss_check is not None:
        new_loss_check = np.mean(losses[-100:])
        # If avg loss increased or plateaued then reduce LR
        if new_loss_check >= loss_check:
            for g in torch.optim.param_groups:
                g['lr'] *= 0.5
        loss_check = new_loss_check

    elif args.lr_plateau and loss_check is None and len(losses) >= 100:
        loss_check = np.mean(losses[-100:])


def export_final_results(args, dir, losses, mesh, mlp, network_input, vertices):
    with torch.no_grad():
        pred_rgb, pred_normal = mlp(network_input)
        pred_rgb = pred_rgb.detach().cpu()
        pred_normal = pred_normal.detach().cpu()

        torch.save(pred_rgb, os.path.join(dir, f"colors_final.pt"))
        torch.save(pred_normal, os.path.join(dir, f"normals_final.pt"))

        base_color = torch.full(size=(mesh.vertices.shape[0], 3), fill_value=0.5)
        final_color = torch.clamp(pred_rgb + base_color, 0, 1)

        if args.color_only:
            mesh.vertices = vertices.detach().cpu() + mesh.vertex_normals.detach().cpu() * pred_normal
        else:
            mesh.vertices = vertices.detach().cpu()

        objbase, extension = os.path.splitext(os.path.basename(args.obj_path))
        mesh.export(os.path.join(dir, f"{objbase}_final.obj"), color=final_color)

        # Run renders
        if args.save_render:
            save_rendered_results(args, dir, final_color, mesh)

        # Save final losses
        torch.save(torch.tensor(losses), os.path.join(dir, "losses.pt"))

def export_full_results(args, dir, losses, mesh, full_mesh, mlp, network_input, vertices, old_indice_to_new=None, new_indice_to_old=None, label=None):
    with torch.no_grad():
        pred_rgb, pred_normal = mlp(network_input)
        pred_rgb = pred_rgb.detach().cpu()
        pred_normal = pred_normal.detach().cpu()

        #torch.save(pred_rgb, os.path.join(dir, f"colors_final.pt"))
        #torch.save(pred_normal, os.path.join(dir, f"normals_final.pt"))

        # if args.with_prior_color:
        #     base_color = mesh.colors.detach().cpu()
        # else:
        #     base_color = torch.full(size=(mesh.vertices.shape[0], 3), fill_value=0.5)
        # final_color = torch.clamp(pred_rgb + base_color, 0, 1)

        #  export train result with only one label, because in this version mesh is full_mesh
        #mesh.vertices = vertices.detach().cpu() + mesh.vertex_normals.detach().cpu() * pred_normal

        #objbase, extension = os.path.splitext(os.path.basename(args.obj_path))
        #mesh.export(os.path.join(dir, f"{label}_{objbase}_final.obj"), color=final_color)

        # if args.with_prior_color:
        #     full_base_color = full_mesh.colors.detach().cpu()
        # else:
        #     full_base_color = torch.full(size=(full_mesh.vertices.shape[0], 3), fill_value=0.5)
        full_pred_rgb = torch.zeros([full_mesh.vertices.shape[0], 3], dtype=torch.float32)
        full_pred_normal = torch.zeros([full_mesh.vertices.shape[0], 1], dtype=torch.float32)

        if args.render_all_grad_one:
            full_pred_rgb = pred_rgb
            full_pred_normal = pred_normal
        elif args.focus_one_thing:
            for old, new in enumerate(old_indice_to_new):
                if new != -1:
                    full_pred_rgb[old] = pred_rgb[new]
                    full_pred_normal[old] = pred_normal[new]

        #full_final_color = torch.clamp(full_pred_rgb + full_base_color, 0, 1)

        # FixMe: input vertices should be fixed
        #full_mesh.vertices = full_mesh.vertices.detach().cpu() + full_mesh.vertex_normals.detach().cpu() * full_pred_normal

        #objbase, extension = os.path.splitext(os.path.basename(args.obj_path))
        #full_mesh.export(os.path.join(dir, f"{label}_{objbase}_full_final.obj"), color=full_final_color)

        # Run renders
        # if args.save_render:
        #     save_rendered_results(args, dir, final_color, mesh)

        # Save final losses
        #torch.save(torch.tensor(losses), os.path.join(dir, "losses.pt"))
        return full_pred_rgb, full_pred_normal


def save_rendered_results(args, dir, final_color, mesh):
    default_color = torch.full(size=(mesh.vertices.shape[0], 3), fill_value=0.5, device=device)
    mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(default_color.unsqueeze(0),
                                                                   mesh.faces.to(device))
    kal_render = Renderer(
        camera=kal.render.camera.generate_perspective_projection(np.pi / 4, 1280 / 720).to(device),
        dim=(1280, 720))
    MeshNormalizer(mesh)()
    img, mask = kal_render.render_single_view(mesh, args.frontview_center[1], args.frontview_center[0],
                                              radius=2.5,
                                              background=torch.tensor([1, 1, 1]).to(device).float(),
                                              return_mask=True)
    img = img[0].cpu()
    mask = mask[0].cpu()
    # Manually add alpha channel using background color
    alpha = torch.ones(img.shape[1], img.shape[2])
    alpha[torch.where(mask == 0)] = 0
    img = torch.cat((img, alpha.unsqueeze(0)), dim=0)
    img = transforms.ToPILImage()(img)
    img.save(os.path.join(dir, f"init_cluster.png"))
    MeshNormalizer(mesh)()
    # Vertex colorings
    mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(final_color.unsqueeze(0).to(device),
                                                                   mesh.faces.to(device))
    img, mask = kal_render.render_single_view(mesh, args.frontview_center[1], args.frontview_center[0],
                                              radius=2.5,
                                              background=torch.tensor([1, 1, 1]).to(device).float(),
                                              return_mask=True)
    img = img[0].cpu()
    mask = mask[0].cpu()
    # Manually add alpha channel using background color
    alpha = torch.ones(img.shape[1], img.shape[2])
    alpha[torch.where(mask == 0)] = 0
    img = torch.cat((img, alpha.unsqueeze(0)), dim=0)
    img = transforms.ToPILImage()(img)
    img.save(os.path.join(dir, f"final_cluster.png"))


def update_mesh(args, mlp, network_input, prior_color, sampled_mesh, vertices, ver_mask=None):
    pred_rgb, pred_normal = mlp(network_input)

    if args.render_all_grad_one:
        pred_rgb = pred_rgb*ver_mask.unsqueeze(dim=-1)
        pred_normal = pred_normal*ver_mask.unsqueeze(dim=-1)

    # sampled_mesh refers to the focused thing
    sampled_mesh.face_attributes = prior_color + kaolin.ops.mesh.index_vertices_by_faces(
        pred_rgb.unsqueeze(0),
        sampled_mesh.faces)
    if args.color_only:
        sampled_mesh.vertices = vertices
    else:
        sampled_mesh.vertices = vertices + sampled_mesh.vertex_normals * pred_normal
    MeshNormalizer(sampled_mesh)()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_path', type=str, default='meshes/mesh1.obj')
    parser.add_argument('--prompt', nargs="+", default='a pig with pants', help='some prompts separated by commas, like \'icy wall, wooden chair, marble floor\', remember you should add the corresponding label number to args.label')
    parser.add_argument('--normprompt', nargs="+", default=None)
    parser.add_argument('--promptlist', nargs="+", default=None)
    parser.add_argument('--normpromptlist', nargs="+", default=None)
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='round2/alpha5')
    parser.add_argument('--traintype', type=str, default="shared")
    parser.add_argument('--sigma', type=float, default=10.0)
    parser.add_argument('--normsigma', type=float, default=10.0)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--colordepth', type=int, default=2)
    parser.add_argument('--normdepth', type=int, default=2)
    parser.add_argument('--normwidth', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--normal_learning_rate', type=float, default=0.0005)
    parser.add_argument('--decay', type=float, default=0)
    parser.add_argument('--lr_decay', type=float, default=1)
    parser.add_argument('--lr_plateau', action='store_true')
    parser.add_argument('--no_pe', dest='pe', default=True, action='store_false')
    parser.add_argument('--decay_step', type=int, default=100)
    parser.add_argument('--n_views', type=int, default=5)
    parser.add_argument('--n_augs', type=int, default=0)
    parser.add_argument('--n_normaugs', type=int, default=0)
    parser.add_argument('--n_iter', type=int, default=6000)
    parser.add_argument('--encoding', type=str, default='gaussian')
    parser.add_argument('--normencoding', type=str, default='xyz')
    parser.add_argument('--layernorm', action="store_true")
    parser.add_argument('--run', type=str, default=None)
    parser.add_argument('--gen', action='store_true')
    parser.add_argument('--clamp', type=str, default="tanh")
    parser.add_argument('--normclamp', type=str, default="tanh")
    parser.add_argument('--normratio', type=float, default=0.1)
    parser.add_argument('--frontview', action='store_true')
    parser.add_argument('--no_prompt', default=False, action='store_true')
    parser.add_argument('--exclude', type=int, default=0)

    parser.add_argument('--frontview_std', type=float, default=8)
    parser.add_argument('--frontview_center', nargs=2, type=float, default=[0., 0.])
    parser.add_argument('--clipavg', type=str, default=None)
    parser.add_argument('--geoloss', action="store_true")
    parser.add_argument('--samplebary', action="store_true")
    parser.add_argument('--promptviews', nargs="+", default=None)
    parser.add_argument('--mincrop', type=float, default=1)
    parser.add_argument('--maxcrop', type=float, default=1)
    parser.add_argument('--normmincrop', type=float, default=0.1)
    parser.add_argument('--normmaxcrop', type=float, default=0.1)
    parser.add_argument('--splitnormloss', action="store_true")
    parser.add_argument('--splitcolorloss', action="store_true")
    parser.add_argument("--nonorm", action="store_true")
    parser.add_argument('--cropsteps', type=int, default=0)
    parser.add_argument('--cropforward', action='store_true')
    parser.add_argument('--cropdecay', type=float, default=1.0)
    parser.add_argument('--decayfreq', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--background', nargs=3, type=float, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_render', action="store_true")
    parser.add_argument('--input_normals', default=False, action='store_true')
    parser.add_argument('--symmetry', default=False, action='store_true')
    parser.add_argument('--only_z', default=False, action='store_true')
    parser.add_argument('--standardize', default=False, action='store_true')
    parser.add_argument('--rand_background', default=False, action='store_true')
    parser.add_argument('--lighting', default=False, action='store_true')
    parser.add_argument('--color_only', default=False, action='store_true', help='only change mesh color instead of changing both color and vertices\' place')
    parser.add_argument('--with_prior_color', default=False, action='store_true', help='render the mesh with its previous color instead of RGB(0.5, 0.5, 0.5)*255')
    parser.add_argument('--label', nargs='+', type=int, default=5, help='need to correspond to the prompt one by one, can read label2class_help.txt to look for labels to class names')
    parser.add_argument('--focus_one_thing', default=False, action='store_true', help='focus on at each rendering vertices/faces with specified label instead of full mesh')
    parser.add_argument('--render_all_grad_one', default=False, action='store_true', help='use full mesh to render, while only change vertices/faces with specified label, must be used with arg.focus_one_thing')
    parser.add_argument('--rand_focal', default=False, action='store_true', help='make carema focal lenth change randomly at each rendering')
    parser.add_argument('--with_hsv_loss', default=False, action='store_true', help='add hsv loss to the loss function')

    # TODO add help for key options
    # args = parser.parse_args()
    args = parse_args()

    run_branched(args)


# For comparison: 10*scenes
# 1. w/ w/o HSV regularization
# 2. full house / part rendering
# 3. random / fixed focal lengths 
# 4. w/ or w/o initial colors
# +5. w/ or w/o semantic mask

# Future 
# 1. full house regularization
# 2. camera pose
# 3. feature interpolation 