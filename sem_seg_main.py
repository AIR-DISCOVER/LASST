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
    ################ Preparing ################
    dir = args.output_dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.seed != 0:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    objbase, _ = os.path.splitext(os.path.basename(args.obj_path))

    ################# Loading #################
    render = Renderer()
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
            cropiter = round(args.n_iter / (args.cropsteps + 1))
            # cropupdate = (args.maxcrop - args.mincrop) / cropiter
            cropupdate = -0.9

            if not args.cropforward:
                cropupdate *= -1

        # Displacement-only augmentations
        displaugment_transform = transforms.Compose(
            [transforms.RandomResizedCrop(224, scale=(args.normmincrop, args.normmincrop)),
             transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5), clip_normalizer])

        normweight = 1.0
        ####################################################

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
                               input_dim=input_dim,
                               exclude=args.exclude).to(device)
        mlp.reset_weights()

        optim = torch.optim.Adam(mlp.parameters(), args.learning_rate, weight_decay=args.decay)
        activate_scheduler = args.lr_decay < 1 and args.decay_step > 0 and args.lr_plateau is None
        if activate_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.decay_step, gamma=args.lr_decay)
        loss_check = None  # For plateau scheduling
        ####################################################

        assert args.prompt is not None or args.image is not None
        if args.prompt is not None:
            prompt = ' '.join(args.prompt)
            prompt = prompt.split(',')[label_order].strip()
            with torch.no_grad():
                prompt_token = clip.tokenize([prompt]).to(device)
                encoded_text = clip_model.encode_text(prompt_token)

            # Save prompt
            with open(os.path.join(dir, prompt), "w") as f:
                f.write("")
            with open(os.path.join(dir, 'prompt.txt'), "a") as f:
                f.write(f"{prompt}\n")

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
            output_mesh.face_attributes = mesh.face_attributes + kaolin.ops.mesh.index_vertices_by_faces(pred_rgb.unsqueeze(0), mesh.faces)
            output_mesh.colors = mesh.colors + pred_rgb
            if not args.color_only:
                output_mesh.vertices = mesh.vertices + mesh.vertex_normals * pred_normal
            MeshNormalizer(output_mesh)()

            loss = 0.0

            ###################### HSV Loss #####################
            if args.with_hsv_loss:
                hsv_loss = 0.0
                if args.render_one_grad_one:
                    h1, s1, v1 = HSV().get_hsv(output_mesh.colors.unsqueeze(-1).unsqueeze(-1))
                    h2, s2, v2 = HSV().get_hsv(mesh.colors.unsqueeze(-1).unsqueeze(-1))
                    h3, s3, v3 = HSV().get_hsv(init_mesh.colors.unsqueeze(-1).unsqueeze(-1))  # TODO
                if args.render_all_grad_one:
                    h1, s1, v1 = HSV().get_hsv(output_mesh.colors[ver_mask].unsqueeze(-1).unsqueeze(-1))
                    h2, s2, v2 = HSV().get_hsv(mesh.colors[ver_mask].unsqueeze(-1).unsqueeze(-1))
                    h3, s3, v3 = HSV().get_hsv(init_mesh.colors[ver_mask].unsqueeze(-1).unsqueeze(-1))  # TODO

                hsv_loss += 0.1 * torch.min((h1 - h2).abs(), 1 - ((h1 - h2).abs())).mean()
                hsv_loss += 0.1 * (s1 - s2).abs().mean()
                hsv_loss += 0.1 * (v1 - v2).abs().mean()
                hsv_loss += 0.5 * torch.min((h1.mean() - h2.mean()).abs(), 1 - ((h1.mean() - h2.mean()).abs()))
                hsv_loss += 0.5 * (s1.mean() - s2.mean()).abs()
                hsv_loss += 0.5 * (v1.mean() - v2.mean()).abs()
                hsv_loss += torch.min((h1.std() - h2.std()).abs(), 1 - ((h1.std() - h2.std()).abs()))
                hsv_loss += (s1.std() - s2.std()).abs()
                hsv_loss += (v1.std() - v2.std()).abs()
                loss += hsv_loss.reshape(1) / 9

            ###################### Render Loss #########################
            # Render output result, use only mesh.vertices, mesh.faces, mesh.face_attributes
            rendered_images, elev, azim = render.render_center_out_views(output_mesh,
                                                                         num_views=args.n_views,
                                                                         lighting=args.lighting,
                                                                         show=args.show,
                                                                         center_azim=args.frontview_center[0],
                                                                         center_elev=args.frontview_center[1],
                                                                         std=args.frontview_std,
                                                                         return_views=True,
                                                                         background=torch.tensor(args.background).to(device),
                                                                         rand_background=args.rand_background,
                                                                         rand_focal=args.rand_focal)

            if args.n_augs == 0:
                clip_image = clip_transform(rendered_images)
                encoded_renders = clip_model.encode_image(clip_image)
                loss -= torch.mean(torch.cosine_similarity(encoded_renders, encoded_text))

            # Check augmentation steps
            if args.cropsteps != 0 and cropupdate != 0 and i != 0 and i % args.cropsteps == 0:
                curcrop += cropupdate
                # print(curcrop)
                normaugment_transform = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(curcrop, curcrop)),
                    transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
                    clip_normalizer,
                ])

            if args.n_augs > 0:
                for _ in range(args.n_augs):
                    augmented_image = augment_transform(rendered_images)
                    encoded_renders = clip_model.encode_image(augmented_image)
                    if args.prompt:
                        if args.clipavg == "view":
                            if encoded_text.shape[0] > 1:
                                loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0), torch.mean(encoded_text, dim=0), dim=0)
                            else:
                                loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True), encoded_text)
                                # embed()
                        else:
                            loss -= torch.mean(torch.cosine_similarity(encoded_renders, encoded_text))
                    if args.image:
                        if encoded_image.shape[0] > 1:
                            loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0), torch.mean(encoded_image, dim=0), dim=0)
                        else:
                            loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True), encoded_image)
                        # if args.image:
                        #     loss -= torch.mean(torch.cosine_similarity(encoded_renders,encoded_image))
            if args.splitnormloss:
                for param in mlp.mlp_normal.parameters():
                    param.requires_grad = False
            if args.splitcolorloss:
                for param in mlp.mlp_rgb.parameters():
                    param.requires_grad = False

            # Normal augment transform
            # unused
            # loss = 0.0
            if args.n_normaugs > 0:
                # loss = 0.0
                for _ in range(args.n_normaugs):
                    augmented_image = normaugment_transform(rendered_images)
                    encoded_renders = clip_model.encode_image(augmented_image)
                    if args.prompt:
                        if args.clipavg == "view":
                            if encoded_text.shape[0] > 1:
                                loss -= normweight * torch.cosine_similarity(torch.mean(encoded_renders, dim=0), torch.mean(encoded_text, dim=0), dim=0)
                            else:
                                loss -= normweight * torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True), encoded_text)
                        else:
                            loss -= normweight * torch.mean(torch.cosine_similarity(encoded_renders, encoded_text))
                    if args.image:
                        if encoded_image.shape[0] > 1:
                            loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0), torch.mean(encoded_image, dim=0), dim=0)
                        else:
                            loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True), encoded_image)
                        # if args.image:
                        #     loss -= torch.mean(torch.cosine_similarity(encoded_renders,encoded_image))
                if args.splitnormloss:
                    for param in mlp.mlp_normal.parameters():
                        param.requires_grad = True
                if args.splitcolorloss:
                    for param in mlp.mlp_rgb.parameters():
                        param.requires_grad = False

            loss.backward(retain_graph=True)

            # Also run separate loss on the uncolored displacements
            if args.geoloss:
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
                    normloss.backward(retain_graph=True)

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

            # Adjust normweight if set
            if args.decayfreq is not None:
                if i % args.decayfreq == 0:
                    normweight *= args.cropdecay

            if i % args.report_step == 0:
                report_process(dir, i, loss, rendered_images, label)

        # full_pred_rgb = torch.zeros([init_mesh.vertices.shape[0], 3], dtype=torch.float32)
        # full_pred_vertices = torch.zeros([init_mesh.vertices.shape[0], 3], dtype=torch.float32)
        # full_final_mask = torch.zeros([init_mesh.vertices.shape[0]], dtype=torch.float32)

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
    init_mesh.export(os.path.join(dir, f"all_{objbase}_full_final.obj"), color=init_mesh.colors)


def report_process(dir, i, loss, rendered_images, label):
    print('iter: {} loss: {}'.format(i, loss.item()))
    torchvision.utils.save_image(rendered_images, os.path.join(dir, 'label_{}_iter_{}.jpg'.format(label, i)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # =================      Input and Output      =================
    parser.add_argument('--obj_path', type=str, default='', help='Obj name w/o .obj suffix')
    parser.add_argument('--label', nargs='+', type=int, default=5, help='indices for semantic categories, joined by space')
    parser.add_argument('--prompt', nargs="+", default='a pig with pants', help='text description for each category, joined by comma. Number of categories should comply with --label')
    parser.add_argument('--image', type=str, default=None)  # TODO
    parser.add_argument('--output_dir', type=str, default='round2/alpha5', help="Output directory")
    # ==============================================================

    # ================= Neural Style Field options =================
    parser.add_argument('--sigma', type=float, default=10.0, help='Neural Style Field: sigma value in gaussian encoder')
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
    # ==============================================================

    # =================   Optimizer and Scheduler  =================
    parser.add_argument('--learning_rate', type=float, default=0.0005, help="Adam Optimizer: learning rate")
    parser.add_argument('--decay', type=float, default=0, help='Adam Optimizer: weight decay')
    parser.add_argument('--lr_decay', type=float, default=1, help='StepLR Scheduler: learning rate decay')
    parser.add_argument('--decay_step', type=int, default=100, help='StepLR Scheduler: decay step')
    parser.add_argument('--n_iter', type=int, default=6000, help="Number of optimizing iterations for each run")
    parser.add_argument('--lr_plateau', type=int, default=None, help="The step of Plateau scheduling (if adopted)")  # FIXME
    # ==============================================================

    # =================           Renderer         =================
    parser.add_argument('--n_views', type=int, default=5, help="Renderer: Number of rendered views")
    parser.add_argument('--frontview_center', nargs=2, type=float, default=[0., 0.], help="Renderer: frontview center")
    parser.add_argument('--frontview_std', type=float, default=8, help="Renderer: frontview standard deviation")
    parser.add_argument('--show', action='store_true', help="Renderer: show with matplotlib when rendering")
    parser.add_argument('--background', nargs=3, type=float, default=None, help='Renderer: base color of background')
    parser.add_argument('--rand_background', default=False, action='store_true', help='Renderer: using randomly point-wise distorted colors as background')
    parser.add_argument('--lighting', default=False, action='store_true', help='Renderer: use lighting and cast shadows')
    parser.add_argument('--mincrop', type=float, default=1)
    parser.add_argument('--maxcrop', type=float, default=1)
    # ==============================================================

    # =================         Preprocess         =================
    parser.add_argument('--input_normals', default=False, action='store_true', help='Preprocess: input points with their normals')
    parser.add_argument('--only_z', default=False, action='store_true', help='Preprocess: input points with z coords only')
    # ==============================================================

    # =================            Misc            =================
    parser.add_argument('--seed', type=int, default=0)
    # ==============================================================

    parser.add_argument('--n_augs', type=int, default=0, help="")  # FIXME
    parser.add_argument('--n_normaugs', type=int, default=0, help="")  # FIXME

    parser.add_argument('--clipavg', type=str, default=None)  # FIXME
    parser.add_argument('--normmincrop', type=float, default=0.1)  # TODO inspection needed
    parser.add_argument('--normmaxcrop', type=float, default=0.1)  # TODO inspection needed
    parser.add_argument('--cropsteps', type=int, default=0)  # TODO
    parser.add_argument('--cropforward', action='store_true')  # TODO
    parser.add_argument('--cropdecay', type=float, default=1.0)  # TODO
    parser.add_argument('--save_render', action="store_true")  # TODO inspection needed

    parser.add_argument('--geoloss', action="store_true", help="Additional loss for displacement")  # TODO
    parser.add_argument('--splitnormloss', action="store_true", help="Displacement loss only backward to displacement head")
    parser.add_argument('--splitcolorloss', action="store_true", help="Displacement loss only backward to displacement head")

    parser.add_argument('--decayfreq', type=int, default=None)  # FIXME loss weight decay, remove it
    # parser.add_argument('--overwrite', action='store_true') # TODO check behavior incase of overwrite

    parser.add_argument('--color_only', default=False, action='store_true', help='only change mesh color instead of changing both color and vertices\' place')
    parser.add_argument('--with_prior_color', default=False, action='store_true', help='render the mesh with its previous color instead of RGB(0.5, 0.5, 0.5)*255')
    parser.add_argument('--render_one_grad_one', default=False, action='store_true', help='focus on at each rendering vertices/faces with specified label instead of full mesh')
    parser.add_argument('--render_all_grad_one',
                        default=False,
                        action='store_true',
                        help='use full mesh to render, while only change vertices/faces with specified label, must be used with arg.render_one_grad_one')
    parser.add_argument('--rand_focal', default=False, action='store_true', help='make carema focal lenth change randomly at each rendering')
    parser.add_argument('--with_hsv_loss', default=False, action='store_true', help='add hsv loss to the loss function')
    parser.add_argument('--regress', default=False, action="store_true")
    parser.add_argument('--report_step', default=100, type=int, help='')

    # TODO add help for key options

    args = parser.parse_args()

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
