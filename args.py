import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_path', type=str, default='meshes/mesh1.obj')
    parser.add_argument('--prompt', nargs="+", default='a pig with pants')
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
    parser.add_argument('--color_only', default=False, action='store_true')
    parser.add_argument('--with_prior_color', default=False, action='store_true')
    parser.add_argument('--label', nargs='+', type=int, default=5)
    parser.add_argument('--render_all_grad_one', default=False, action='store_true')
    parser.add_argument('--focus_one_thing', default=False, action='store_true')

    # comparison options
    parser.add_argument('--hsv_diff_constraints', action='store_true', help="Apply constraint loss: absolute changes of pixels' hsv")
    parser.add_argument('--hsv_diff_coefficient',
                        nargs=3,
                        type=float,
                        default=[1., 1., 1.],
                        help="The coefficients of H,S,V Diff loss respectively")
    parser.add_argument('--hsv_stat_constraints',
                        action='store_true',
                        help="Apply constraint loss: absolute changes of hsv average and standard deviation")
    parser.add_argument('--hsv_mean_diff_coefficient',
                        nargs=3,
                        type=float,
                        default=[1., 1., 1.],
                        help="The coefficients of H,S,V Diff loss respectively")
    parser.add_argument('--hsv_std_diff_coefficient',
                        nargs=3,
                        type=float,
                        default=[1., 1., 1.],
                        help="The coefficients of H,S,V Diff loss respectively")
    parser.add_argument('--rand_focal', action='store_true', help="Apply constraints: random focal lengths")

    return parser.parse_args()