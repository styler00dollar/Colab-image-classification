from ffcv.fields import RGBImageField

from ffcv_pl.generate_dataset import create_beton_wrapper
import yaml

with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)


def main():
    from data import ImageDataloader

    image_label_dataset = ImageDataloader(
        data_root=cfg["path"]["training_path"],
        size=cfg["size"],
        means=cfg["means"],
        std=cfg["std"],
        ffcv=True,
    )

    fields = (
        RGBImageField(write_mode="jpg", jpeg_quality=95, max_resolution=512),
        None,
    )
    create_beton_wrapper(image_label_dataset, "./data/images.beton", fields)


if __name__ == "__main__":

    main()
