import clip
from tqdm import tqdm
import kaolin.ops.mesh
import torch
from args import parse_args
from neural_style_field import NeuralStyleField
from utils import device
from render import Renderer
from mesh_scannet import Mesh
from utils import clip_model
from Normalization import MeshNormalizer
from utils import preprocess
import numpy as np
import random
import torchvision
import os
from PIL import Image
import argparse
from pathlib import Path
from torchvision import transforms
from convert import HSVLoss as HSV
import json
from IPython import embed

if __name__ == '__main__':
    print('imported')

# 1. camera position
# 2. H of HSV range
# 3. random camera pose until fore ground larger that 80%
# 4. avoid faces


def run(args):
    ################    Seed   ################
    if args.seed != 0:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    ################# Loading #################
    render = Renderer()  # pi / 3
    init_mesh = Mesh(args.obj_path)
    # Bounding sphere normalizer
    # mesh.vertices is modified
    MeshNormalizer(init_mesh)()

    full_pred_rgb = init_mesh.colors.detach()
    full_pred_vertices = init_mesh.vertices.detach()

    for label_order, label in enumerate(args.label):
        if not (init_mesh.labels == label).any():
            print(f"label {label} is not in this mesh")
            continue

        # Set up index mapping
        ver_mask, face_mask, old_indice_to_new, new_indice_to_old = init_mesh.get_mask(label)

        # Create a full copy of initial mesh
        assert not (args.render_one_grad_one and args.render_all_grad_one)
        if args.render_one_grad_one:
            mesh = init_mesh.mask_mesh(ver_mask, face_mask, old_indice_to_new, new_indice_to_old)
        else:
            mesh = init_mesh.clone()

        if not args.with_prior_color:
            mesh.colors = torch.full(size=(mesh.colors.shape[0], 3), fill_value=0.5, device=device)
            mesh.face_attributes = torch.full(size=(mesh.faces.shape[0], 3, 3), fill_value=0.5, device=device)
        render_args = []
        fail = False
        for i in range(args.n_views):
            result = render.find_appropriate_view(mesh, args.view_min, args.view_max, percent=1 / (i + 1))
            if result is None:
                fail = True
                break
            render_args.append(result)
        if fail:
            continue

        losses = []

        #################### Transforms ####################
        clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        # CLIP Transform
        clip_transform = transforms.Compose([transforms.Resize((224, 224)), clip_normalizer])

        # Augmentation settings
        augment_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(args.mincrop, args.maxcrop)),
            transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
            clip_normalizer,
        ])

        # Augmentations for normal network
        if args.cropforward:
            curcrop = args.normmincrop
        else:
            curcrop = args.normmaxcrop
        normaugment_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(curcrop, curcrop)),
            transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
            clip_normalizer,
        ])
        cropiter = 0
        cropupdate = 0
        if args.normmincrop < args.normmaxcrop and args.cropsteps > 0:
            cropiter = round((args.n_iter + args.cropsteps - 1) // args.cropsteps)
            cropupdate = (args.normmaxcrop - args.normmincrop) / cropiter

            if not args.cropforward:
                cropupdate *= -1

        # Displacement-only augmentations
        displaugment_transform = transforms.Compose(
            [transforms.RandomResizedCrop(224, scale=(args.normmincrop, args.normmincrop)),
             transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5), clip_normalizer])

        norm_loss_weight = 1.0

        ################### MLP Settings ###################
        if args.only_z:
            input_dim = 1
        elif args.input_normals:
            input_dim = 6
        else:
            input_dim = 3
        mlp = NeuralStyleField(args.sigma,
                               args.depth,
                               args.width,
                               'gaussian',
                               args.colordepth,
                               args.normdepth,
                               args.normratio,
                               args.clamp,
                               args.normclamp,
                               niter=args.n_iter,
                               progressive_encoding=args.pe,
                               input_dim=input_dim).to(device)
        mlp.reset_weights()

        optim = torch.optim.Adam(mlp.parameters(), args.learning_rate, weight_decay=args.decay)
        activate_scheduler = args.lr_decay < 1 and args.decay_step > 0 and args.lr_plateau is None
        if activate_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.decay_step, gamma=args.lr_decay)
        loss_check = None  # For plateau scheduling

        assert args.prompt is not None or args.image is not None
        if args.prompt is not None:
            prompt = ' '.join(args.prompt)
            prompt = prompt.split(',')[label_order].strip()
            with torch.no_grad():
                prompt_token = clip.tokenize([prompt]).to(device)
                encodedtext = clip_model.encode_text(prompt_token)

            # Save prompt
            with open(os.path.join(args.dir, f"1prompt-{prompt}"), "w") as f:
                f.write("")
        if args.forbidden is not None:
            forbidden = ' '.join(args.forbidden)
            forbidden = [i.strip() for i in forbidden.split(',')]
            with torch.no_grad():
                forbidden_token = clip.tokenize(forbidden).to(device)
                encoded_forbidden = clip_model.encode_text(forbidden_token)

            # Save prompt
            with open(os.path.join(args.dir, f"2forbidden-{forbidden}"), "w") as f:
                f.write("")

        if args.image is not None:
            img = Image.open(args.image)
            with torch.no_grad():
                img = preprocess(img).to(device)
                encoded_image = clip_model.encode_image(img.unsqueeze(0))

        for i in tqdm(range(args.n_iter)):
            optim.zero_grad()

            # only coordinates?
            pred_rgb, pred_normal = mlp(mesh.vertices)

            if args.render_all_grad_one:
                pred_rgb = pred_rgb * (ver_mask.to(torch.long)).unsqueeze(dim=-1)
                pred_normal = pred_normal * (ver_mask.to(torch.long)).unsqueeze(dim=-1)

            output_mesh = mesh.clone()
            output_mesh.face_attributes = torch.clamp(mesh.face_attributes + kaolin.ops.mesh.index_vertices_by_faces(pred_rgb.unsqueeze(0), mesh.faces), 0, 1)
            output_mesh.colors = torch.clamp(mesh.colors + pred_rgb, 0, 1)
            if not args.color_only:
                output_mesh.vertices = mesh.vertices + mesh.vertex_normals * pred_normal
            MeshNormalizer(output_mesh)()

            loss = torch.tensor([0.]).cuda()
            hsv_loss = torch.tensor([0.]).cuda()
            hsv_stat_loss = torch.tensor([0.]).cuda()
            sv_stat_loss = torch.tensor([0.]).cuda()
            rgb_loss = torch.tensor([0.]).cuda()
            if args.rgb_loss_weight is not None:
                if args.render_one_grad_one:
                    rgb_loss += args.rgb_loss_weight * (torch.abs(output_mesh.colors.flatten() - mesh.colors.flatten())).mean()
                if args.render_all_grad_one:
                    rgb_loss += args.rgb_loss_weight * (torch.abs(output_mesh.colors[ver_mask].flatten() - mesh.colors[ver_mask].flatten())).mean()

            ###################### HSV Loss #####################
            if args.render_one_grad_one:
                h1, s1, v1 = HSV().get_hsv(output_mesh.colors.unsqueeze(-1).unsqueeze(-1))
                h2, s2, v2 = HSV().get_hsv(mesh.colors.unsqueeze(-1).unsqueeze(-1))
                # h3, s3, v3 = HSV().get_hsv(init_mesh.colors.unsqueeze(-1).unsqueeze(-1))  # TODO
            if args.render_all_grad_one:
                h1, s1, v1 = HSV().get_hsv(output_mesh.colors[ver_mask].unsqueeze(-1).unsqueeze(-1))
                h2, s2, v2 = HSV().get_hsv(mesh.colors[ver_mask].unsqueeze(-1).unsqueeze(-1))
                # h3, s3, v3 = HSV().get_hsv(init_mesh.colors[ver_mask].unsqueeze(-1).unsqueeze(-1))  # TODO

            if args.hsv_loss_weight is not None:
                hsv_loss += args.hsv_loss_weight * (torch.min((h1 - h2).abs(), 1 - ((h1 - h2).abs())).mean() + (s1 - s2).abs().mean() + (v1 - v2).abs().mean()) / 3

            if args.sv_stat_loss_weight is not None:
                sv_stat_loss += (s1.mean() - s2.mean()).abs()
                sv_stat_loss += (v1.mean() - v2.mean()).abs()
                sv_stat_loss += (s1.std() - s2.std()).abs()
                sv_stat_loss += (v1.std() - v2.std()).abs()
                sv_stat_loss /= 4
                sv_stat_loss *= args.sv_stat_loss_weight
            if args.hsv_stat_loss_weight is not None:
                h1_vx = torch.cos(2 * np.pi * h1)
                h1_vy = torch.sin(2 * np.pi * h1)
                h1_v = torch.stack((h1_vx, h1_vy), -1)
                h2_vx = torch.cos(2 * np.pi * h2)
                h2_vy = torch.sin(2 * np.pi * h2)
                h2_v = torch.stack((h2_vx, h2_vy), -1)
                diff = (h1_v - h2_v).squeeze().mean(dim=0)
                hsv_stat_loss += args.hsv_stat_loss_weight * torch.linalg.norm(diff).mean()
                # hsv_loss += 0.5 * torch.min((h1.mean() - h2.mean()).abs(), 1 - ((h1.mean() - h2.mean()).abs()))
                # hsv_loss += 0.5 * (s1.mean() - s2.mean()).abs()
                # hsv_loss += 0.5 * (v1.mean() - v2.mean()).abs()
                # hsv_loss += (s1.std() - s2.std()).abs()
                # hsv_loss += (v1.std() - v2.std()).abs()
                # hsv_loss += torch.min((h1.std() - h2.std()).abs(), 1 - ((h1.std() - h2.std()).abs()))
                #loss += hsv_loss.reshape(1) / 9

            ###################### Render Loss #########################
            # Render output result, use only mesh.vertices, mesh.faces, mesh.face_attributes
            rendered_images, elev, azim = render.render_center_out_views(
                output_mesh,
                num_views=args.n_views,
                lighting=args.lighting,
                show=args.show,
                render_args=render_args,
                elev_std=args.frontview_elev_std,
                azim_std=args.frontview_azim_std,
                return_views=True,
                background=torch.tensor(args.background).to(device),
                rand_background=args.rand_background,
                fixed=args.fixed,
                fixed_all=args.fixed_all,
            )

            text_loss = torch.tensor([0.]).cuda()
            image_loss = torch.tensor([0.]).cuda()
            forbidden_loss = torch.tensor([0.]).cuda()

            if args.n_augs == 0:
                clip_image = clip_transform(rendered_images)
                encoded_renders = clip_model.encode_image(clip_image)
                text_loss -= torch.mean(torch.cosine_similarity(encoded_renders, encoded_text))

            # Check augmentation steps
            if args.cropsteps != 0 and cropupdate != 0 and i != 0 and i % args.cropsteps == 0:
                curcrop += cropupdate
                normaugment_transform = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(curcrop, curcrop)),
                    transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
                    clip_normalizer,
                ])

            if args.n_augs > 0:
                for _ in range(args.n_augs):
                    augmented_image = augment_transform(rendered_images)
                    encoded_renders = clip_model.encode_image(augmented_image)
                    if args.prompt is not None:
                        if args.clipavg:
                            if encoded_text.shape[0] > 1:
                                text_loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0), torch.mean(encoded_text, dim=0), dim=0)
                            else:
                                text_loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True), encoded_text)
                                # embed()
                        else:
                            text_loss -= torch.mean(torch.cosine_similarity(encoded_renders, encoded_text))
                    if args.forbidden is not None:
                        if args.clipavg:
                            if encoded_forbidden.shape[0] > 1:
                                forbidden_loss += torch.cosine_similarity(torch.mean(encoded_renders, dim=0), torch.mean(encoded_forbidden, dim=0), dim=0) - 1
                            else:
                                forbidden_loss += torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True), encoded_forbidden) - 1
                                # embed()
                        else:
                            forbidden_loss += torch.mean(torch.cosine_similarity(encoded_renders, encoded_forbidden)) - 1
                    if args.image is not None:
                        if args.clipavg:
                            if encoded_image.shape[0] > 1:
                                image_loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0), torch.mean(encoded_image, dim=0), dim=0)
                            else:
                                image_loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True), encoded_image)
                        else:
                            image_loss -= torch.mean(torch.cosine_similarity(encoded_renders, encoded_image))
                text_loss /= args.n_augs
                image_loss /= args.n_augs
                forbidden_loss /= args.n_augs

            # Normal augment transform
            norm_text_loss = torch.tensor([0.]).cuda()
            norm_image_loss = torch.tensor([0.]).cuda()
            norm_forbidden_loss = torch.tensor([0.]).cuda()
            if args.n_normaugs > 0:
                # loss = 0.0
                for _ in range(args.n_normaugs):
                    augmented_image = normaugment_transform(rendered_images)
                    encoded_renders = clip_model.encode_image(augmented_image)
                    if args.prompt:
                        if args.clipavg:
                            if encoded_text.shape[0] > 1:
                                norm_text_loss -= norm_loss_weight * torch.cosine_similarity(torch.mean(encoded_renders, dim=0), torch.mean(encoded_text, dim=0), dim=0)
                            else:
                                norm_text_loss -= norm_loss_weight * torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True), encoded_text)
                        else:
                            norm_text_loss -= norm_loss_weight * torch.mean(torch.cosine_similarity(encoded_renders, encoded_text))
                    if args.forbidden:
                        if args.clipavg:
                            if encoded_forbidden.shape[0] > 1:
                                norm_forbidden_loss += norm_loss_weight * (torch.cosine_similarity(torch.mean(encoded_renders, dim=0), torch.mean(encoded_forbidden, dim=0), dim=0) - 1)
                            else:
                                norm_forbidden_loss += norm_loss_weight * (torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True), encoded_forbidden) - 1)
                        else:
                            norm_forbidden_loss += norm_loss_weight * (torch.mean(torch.cosine_similarity(encoded_renders, encoded_forbidden)) - 1)
                    if args.image:
                        if args.clipavg:
                            if encoded_image.shape[0] > 1:
                                norm_image_loss -= norm_loss_weight * torch.cosine_similarity(torch.mean(encoded_renders, dim=0), torch.mean(encoded_image, dim=0), dim=0)
                            else:
                                norm_image_loss -= norm_loss_weight * torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True), encoded_image)
                        else:
                            norm_image_loss -= norm_loss_weight * torch.mean(torch.cosine_similarity(encoded_renders, encoded_image))
                norm_text_loss /= args.n_normaugs
                norm_image_loss /= args.n_normaugs
                norm_forbidden_loss /= args.n_normaugs

            # Also run separate loss on the uncolored displacements
            if args.geoloss:
                # FIXME
                default_color = torch.zeros(len(output_mesh.vertices), 3).to(device)
                default_color[:, :] = torch.tensor([0.5, 0.5, 0.5]).to(device)
                output_mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(default_color.unsqueeze(0), output_mesh.faces)
                geo_renders, elev, azim = render.render_center_out_views(output_mesh,
                                                                         num_views=args.n_views,
                                                                         show=args.show,
                                                                         center_azim=args.frontview_center[0],
                                                                         center_elev=args.frontview_center[1],
                                                                         std=args.frontview_std,
                                                                         return_views=True,
                                                                         lighting=True,
                                                                         background=torch.tensor(args.background).to(device))
                if args.n_normaugs > 0:
                    normloss = 0.0
                    ### avgview != aug
                    for _ in range(args.n_normaugs):
                        augmented_image = displaugment_transform(geo_renders)
                        encoded_renders = clip_model.encode_image(augmented_image)
                        if args.prompt:
                            if encoded_text.shape[0] > 1:
                                normloss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0), torch.mean(encoded_text, dim=0), dim=0)
                            else:
                                normloss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True), encoded_text)
                        if args.image:
                            if encoded_image.shape[0] > 1:
                                normloss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0), torch.mean(encoded_image, dim=0), dim=0)
                            else:
                                normloss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True), encoded_image)  # if args.image:
                            #     loss -= torch.mean(torch.cosine_similarity(encoded_renders,encoded_image))
                    # if not args.no_prompt:
                    normloss.backward(retain_graph=args.retain_graph)

            loss += hsv_loss + hsv_stat_loss + sv_stat_loss + rgb_loss + image_loss + text_loss + forbidden_loss + norm_image_loss + norm_text_loss + norm_forbidden_loss
            loss.backward(retain_graph=args.retain_graph)
            optim.step()
            if args.regress:
                mesh = output_mesh
            for param in mlp.mlp_normal.parameters():
                param.requires_grad = True
            for param in mlp.mlp_rgb.parameters():
                param.requires_grad = True

            with torch.no_grad():
                losses.append(loss.item())
            if activate_scheduler:
                lr_scheduler.step()
            if args.lr_plateau is not None and i % args.lr_plateau == 0:
                if loss_check is not None:
                    new_loss_check = np.mean(losses[-args.lr_plateau:])
                    if new_loss_check >= loss_check:
                        for g in torch.optim.param_groups:
                            g['lr'] *= 0.5
                    loss_check = new_loss_check
                elif len(losses >= args.lr_plateau):
                    loss_check = np.mean(losses[-args.lr_plateau:])

            # Adjust norm_loss_weight if set
            if args.norm_loss_decay_freq is not None:
                if i % args.norm_loss_decay_freq == 0:
                    norm_loss_weight *= args.norm_loss_decay

            if i % args.report_step == 0:
                report_process(
                    args.dir, i, loss, rendered_images, label, {
                        'rgb_loss': rgb_loss,
                        'hsv_loss': hsv_loss,
                        'sv_stat_loss': sv_stat_loss,
                        'hsv_stat_loss': hsv_stat_loss,
                        'text_loss': text_loss,
                        'image_loss': image_loss,
                        'forbidden_loss': forbidden_loss,
                        'norm_text_loss': norm_text_loss,
                        'norm_image_loss': norm_image_loss,
                        'norm_forbidden_loss': norm_forbidden_loss,
                    })

        if args.render_one_grad_one:
            full_pred_rgb[ver_mask] = output_mesh.colors
            if not args.color_only:
                full_pred_vertices[ver_mask] = output_mesh.vertices
        elif args.render_all_grad_one:
            full_pred_rgb[ver_mask] = output_mesh.colors[ver_mask]
            if not args.color_only:
                full_pred_vertices[ver_mask] = output_mesh.vertices[ver_mask]
        else:
            raise NotImplementedError

    # FixMe: input vertices should be fixed
    init_mesh.colors = full_pred_rgb
    if not args.color_only:
        init_mesh.vertices = full_pred_vertices

    objbase, extension = os.path.splitext(os.path.basename(args.obj_path))
    init_mesh.export(os.path.join(args.dir, f"all_{objbase}_full_final.obj"), color=init_mesh.colors)


def report_process(dir, i, loss, rendered_images, label, loss_group):
    print('iter: {} loss: {}'.format(i, loss.item()))
    if loss_group is not None:
        print('\t' + '\t'.join([f'{k}:{w.item():2f}' for k, w in loss_group.items()]))
    torchvision.utils.save_image(rendered_images, os.path.join(dir, 'label_{}_iter_{}.jpg'.format(label, i)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # =================      Input and Output      =================
    parser.add_argument('--obj_path', type=str, default='', help='Obj name w/o .obj suffix')
    parser.add_argument('--label', nargs='+', type=int, default=5, help='indices for semantic categories, joined by space')
    parser.add_argument('--prompt', nargs="+", default=None, help='text description for each category, joined by comma. Number of categories should comply with --label')
    parser.add_argument('--forbidden', nargs="+", default=None, help='text description for each category, joined by comma. Number of categories should comply with --label')
    parser.add_argument('--image', type=str, default=None)  # TODO
    parser.add_argument('--output_dir', type=str, default='round2/alpha5', help="Output directory")
    # ==============================================================

    # ================= Neural Style Field options =================
    parser.add_argument('--sigma', type=float, default=5.0, help='Neural Style Field: sigma value in gaussian encoder')
    parser.add_argument('--depth', type=int, default=4, help='Neural Style Field: number of common Linear+ReLU layers')
    parser.add_argument('--width', type=int, default=256, help='Neural Style Field: feature dimensions of common Linear layers')
    parser.add_argument('--colordepth', type=int, default=2, help='Neural Style Field: number of Linear+ReLU layers in color head')
    parser.add_argument('--normdepth', type=int, default=2, help='Neural Style Field: number of Linear+ReLU in displacement head')
    parser.add_argument('--no_pe', dest='pe', default=True, action='store_false', help="Neural Style Field: no progressive encoding")
    parser.add_argument('--clamp', type=str, default="tanh", help="Neural Style Field: clamp method for color")
    parser.add_argument('--normclamp', type=str, default="tanh", help="Neural Style Field: clamp method for displacement")
    parser.add_argument('--normratio', type=float, default=0.1, help="Neural Style Field: Boundaries for displacement")
    parser.add_argument('--encoding', type=str, default='gaussian', help="Neural Style Field: position encoding")
    parser.add_argument('--exclude', type=int, default=0, help="Neural Style Field: UNUSED param in positional encoders")
    parser.add_argument('--splitnormloss', action="store_true", help="Neural Style Field: Loss only backward to displacement head")
    parser.add_argument('--splitcolorloss', action="store_true", help="Neural Style Field: Loss only backward to displacement head")
    # ==============================================================

    # =================   Optimizer and Scheduler  =================
    parser.add_argument('--learning_rate', type=float, default=0.0005, help="Adam Optimizer: learning rate")
    parser.add_argument('--decay', type=float, default=0, help='Adam Optimizer: weight decay')
    parser.add_argument('--lr_decay', type=float, default=1, help='StepLR Scheduler: learning rate decay')
    parser.add_argument('--decay_step', type=int, default=100, help='StepLR Scheduler: decay step')
    parser.add_argument('--n_iter', type=int, default=6000, help="Number of optimizing iterations for each run")
    parser.add_argument('--lr_plateau', type=int, default=None, help="The step of Plateau scheduling (if adopted)")  # FIXME
    parser.add_argument('--retain_graph', default=False, action='store_true', help='retain_graph in loss backward')
    # ==============================================================

    # =================           Renderer         =================
    parser.add_argument('--n_views', type=int, default=5, help="Renderer: Number of rendered views")
    parser.add_argument('--frontview_center', nargs=2, type=float, default=[0., 0.], help="Renderer: frontview center")
    parser.add_argument('--frontview_elev_std', type=float, default=8, help="Renderer: frontview standard deviation")
    parser.add_argument('--frontview_azim_std', type=float, default=8, help="Renderer: frontview standard deviation")
    parser.add_argument('--view_min', type=float, default=0.6, help="Renderer: ")
    parser.add_argument('--view_max', type=float, default=0.9, help="Renderer: ")
    parser.add_argument('--fixed', action='store_true', default=False, help="Renderer: ")
    parser.add_argument('--fixed_all', action='store_true', default=False, help="Renderer: ")
    parser.add_argument('--show', action='store_true', help="Renderer: show with matplotlib when rendering")
    parser.add_argument('--background', nargs=3, type=float, default=None, help='Renderer: base color of background')
    parser.add_argument('--rand_background', default=False, action='store_true', help='Renderer: using randomly point-wise distorted colors as background')
    parser.add_argument('--lighting', default=False, action='store_true', help='Renderer: use lighting and cast shadows')
    parser.add_argument('--rand_focal', default=False, action='store_true', help='make carema focal lenth change randomly at each rendering')
    parser.add_argument('--render_one_grad_one', default=False, action='store_true', help='focus on at each rendering vertices/faces with specified label instead of full mesh')
    parser.add_argument('--render_all_grad_one', default=False, action='store_true', help='use full mesh to render, while only change vertices/faces with specified label')
    parser.add_argument('--with_prior_color', default=False, action='store_true', help='render the mesh with its previous color instead of RGB(0.5, 0.5, 0.5)*255')
    # ==============================================================

    # =================         Preprocess         =================
    parser.add_argument('--input_normals', default=False, action='store_true', help='Preprocess: input points with their normals')
    parser.add_argument('--only_z', default=False, action='store_true', help='Preprocess: input points with z coords only')
    # ==============================================================

    # =================        Augmentation        =================
    parser.add_argument('--n_augs', type=int, default=0, help="Augmentation: Number of augmentation")
    parser.add_argument('--n_normaugs', type=int, default=0, help="Augmentation: Number of normalized augmentation")
    parser.add_argument('--maxcrop', type=float, default=1, help="Augmentation: cropping max range for augmenration")
    parser.add_argument('--mincrop', type=float, default=1, help="Augmentation: cropping min range for augmenration")
    parser.add_argument('--normmincrop', type=float, default=0.1, help="Augmentation: cropping min range for normalized augmenration")
    parser.add_argument('--normmaxcrop', type=float, default=0.1, help="Augmentation: cropping max range for normalized augmenration")
    parser.add_argument('--cropsteps', type=int, default=1, help="Augmentation: step before adjusting normalized augmenration cropping ratio")
    parser.add_argument('--cropforward', action='store_true', help="Augmentation: if true, cropping ratio will be increasing with step instead of decreasing")
    # ==============================================================

    # parser.add_argument('--overwrite', action='store_true') # TODO check behavior incase of overwrite
    # =================            Loss            =================
    parser.add_argument('--geoloss', action="store_true", help="Additional loss for displacement")
    parser.add_argument('--clipavg', action="store_true", default=False, help="view: calculate similarity after calculate mean value")
    parser.add_argument('--hsv_loss_weight', default=None, type=float, help='add hsv loss to the loss function')
    parser.add_argument('--sv_stat_loss_weight', default=None, type=float, help='add hsv loss to the loss function')
    parser.add_argument('--hsv_stat_loss_weight', default=None, type=float, help='add hsv loss to the loss function')
    parser.add_argument('--rgb_loss_weight', default=None, type=float, help='add rgb loss to the loss function')
    parser.add_argument('--color_only', default=False, action='store_true', help='only change mesh color instead of changing both color and vertices\' place')
    parser.add_argument('--norm_loss_decay', type=float, default=1.0)
    parser.add_argument('--norm_loss_decay_freq', type=int, default=None)
    # ==============================================================

    # =================            Misc            =================
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--regress', default=False, action="store_true", help="not ready")  # TODO
    parser.add_argument('--report_step', default=100, type=int, help='The step interval between report calculation')
    parser.add_argument('--dry_run', action="store_true", help="dry run")
    # ==============================================================

    args = parser.parse_args()

    idx = 0
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    while (Path(args.output_dir) / f'{idx}').exists():
        idx += 1
    (Path(args.output_dir) / f'{idx}').mkdir(parents=True, exist_ok=False)
    args.dir = f"{args.output_dir}/{idx}"

    with open(Path(args.output_dir) / f'{idx}' / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    if not args.dry_run:
        run(args)

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
