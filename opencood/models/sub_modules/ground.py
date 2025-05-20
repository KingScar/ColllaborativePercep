import torch
import torch.nn as nn
import torch.nn.functional as F



class BoxEncoding(nn.Module):
    def __init__(self, args):
        super(BoxEncoding, self).__init__()

        self.input_dim = args['boundingbox']
        self.output_dim = args['featuredim']
        self.hidden_dim = args['hiddendim']

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.ln2 = nn.LayerNorm(self.output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        return x



class BEVFeatureGenerator(nn.Module):
    def __init__(self, bev_shape=(200, 200), pos_dim=64, obj_dim=64, mlp_hidden=128, out_dim=64):
        super(BEVFeatureGenerator, self).__init__()
        self.H, self.W = bev_shape
        self.pos_dim = pos_dim
        self.obj_dim = obj_dim
        self.out_dim = out_dim

        pos_encoding = self.get_2d_sin_cos_positional_encoding(self.H, self.W, pos_dim)
        self.register_buffer("pos_encoding", pos_encoding)  # shape: [H, W, pos_dim]

        # MLP to generate grid feature from [pos_encoding + object_feature]
        self.mlp = nn.Sequential(
            nn.Linear(pos_dim + obj_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, out_dim)
        )

    def get_2d_sin_cos_positional_encoding(self, H, W, dim):
        assert dim % 2 == 0
        pe = torch.zeros(H, W, dim)
        y_embed = torch.arange(H).unsqueeze(1).repeat(1, W)
        x_embed = torch.arange(W).unsqueeze(0).repeat(H, 1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(torch.log(torch.tensor(10000.0)) / dim))
        pe[:, :, 0::2] = torch.sin(x_embed.unsqueeze(-1) * div_term)
        pe[:, :, 1::2] = torch.cos(y_embed.unsqueeze(-1) * div_term)
        return pe

    def forward(self, object_grids: list[torch.Tensor], object_features: torch.Tensor):
        """
        object_grids: list of [num_grids_i, 2] (y, x)
        object_features: [num_objects, obj_dim]
        """
        device = self.pos_encoding.device
        bev_feature_map = torch.zeros(self.H, self.W, self.out_dim, device=device)

        # Flatten all grids and record object ids
        all_grids = torch.cat(object_grids, dim=0).to(device)  # [N, 2]
        object_ids = torch.cat([
            torch.full((len(g),), i, dtype=torch.long)
            for i, g in enumerate(object_grids)
        ]).to(device)  # [N]

        y, x = all_grids[:, 0], all_grids[:, 1]
        pos_feats = self.pos_encoding[y, x]  # [N, pos_dim]
        obj_feats = object_features[object_ids]  # [N, obj_dim]

        # concat
        fused_feats = torch.cat([pos_feats, obj_feats], dim=-1)  # [N, pos_dim + obj_dim]
        grid_feats = self.mlp(fused_feats)  # [N, out_dim]

        # encoding into the map
        bev_feature_map[y, x] = grid_feats
        return bev_feature_map  # [H, W, out_dim]



class BEVMapNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.box_encoder = BoxEncoding(
            input_dim=args['boundingbox'],
            hidden_dim=args['hiddendim'],
            output_dim=args['featuredim']
        )
        self.bev_generator = BEVFeatureGenerator(
            bev_shape=(200, 200),
            pos_dim=args['positiondim'],
            obj_dim=args['featuredim'],
            mlp_hidden=args['mlphidden'],
            out_dim=args['outputdim']
        )

    def forward(self, bounding_boxes: torch.Tensor, object_grids: list[torch.Tensor]):
        """
        bounding_boxes: [num_objects, box_dim]
        object_grids: list of [num_grids_i, 2]
        """
        object_feats = self.box_encoder(bounding_boxes)  # [num_objects, obj_dim]
        bev_map = self.bev_generator(object_grids, object_feats)
        return bev_map  # [200, 200, outputdim]


