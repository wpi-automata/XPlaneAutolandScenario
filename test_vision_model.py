import argparse
import torch
import matplotlib.pyplot as plt

from xplane_autoland.vision.perception import AutolandPerceptionModel
from xplane_autoland.vision.xplane_data import AutolandImageDataset


def mse(label, prediction):
    return torch.sum(torch.square(label - prediction)) / label.numel()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a given network on randomly selected data"
    )
    parser.add_argument(
        "--model",
        help="The parameters for the model",
        default="./models/2023-12-6/best_model_params.pt",
    )
    parser.add_argument(
        "--resnet-version",
        help="Which resnet to use",
        choices=["18", "50"],
        default="50",
    )
    parser.add_argument(
        "--data-dir",
        help="Where the data is stored. Expecting images directory and processed-states.csv",
        default="./data",
    )
    parser.add_argument(
        "--seed", help="The random seed for shuffling data", type=int, default=1
    )
    parser.add_argument(
        "--num-samples", help="How many samples to check", type=int, default=10
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    G = torch.Generator()
    G = G.manual_seed(args.seed)

    print(
        f"Loading model from {args.model} with backbone architecture ResNet-{args.resnet_version}"
    )
    model = AutolandPerceptionModel(resnet_version=args.resnet_version)
    model.load(args.model)

    print(f"Loading dataset from: {args.data_dir}")
    # Note: not passing transform=model.preprocess because we save already preprocessed images
    dataset = AutolandImageDataset(
        f"{args.data_dir}/processed-states.csv", f"{args.data_dir}/images"
    )

    sampler = torch.utils.data.RandomSampler(data_source=dataset, generator=G)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler)

    test_iter = iter(testloader)
    for i in range(args.num_samples):
        rwy_img, orient_alt, labels = next(test_iter)
        print("orientation and altitude:", orient_alt)
        print("labels:", labels)
        with torch.no_grad():
            out = model(rwy_img, orient_alt)
        print("prediction:", out)
        print("mse:", mse(labels, out))
        plt.imshow(rwy_img[0].permute(1, 2, 0))
        plt.show()
