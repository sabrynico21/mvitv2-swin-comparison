from argparse import ArgumentParser

def parse_cmd():
    parser = ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Model to train",
        choices=["mvitv2", "swin_t", "swin_s", "swin_b"],
    )
    parser.add_argument(
        "-C",
        "--collation",
        default='trimming',
        help='Choose the collation mode: determine if videos length must be uniformed through padding or trimming',
        choices=["padding", "trimming"]
    )

    args = parser.parse_args()

    match args.model:
        case 'mvitv2':
            from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
            model = mvit_v2_s(weights=MViT_V2_S_Weights.KINETICS400_V1).cuda()
        case 'swin_t':
            from torchvision.models.video import swin3d_t, Swin3D_T_Weights
            model = swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1).cuda()
        case 'swin_s':
            from torchvision.models.video import swin3d_s, Swin3D_S_Weights
            model = swin3d_s(weights=Swin3D_S_Weights.KINETICS400_V1).cuda()
        case 'swin_b':
            from torchvision.models.video import swin3d_b, Swin3D_B_Weights
            model = swin3d_b(weights=Swin3D_B_Weights.KINETICS400_V1).cuda()
        case _:
            raise Exception

    return model, args.collation

