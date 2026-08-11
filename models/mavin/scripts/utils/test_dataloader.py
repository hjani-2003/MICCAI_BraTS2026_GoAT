from monai import data
from monai import transforms

from .datafold_read import datafold_read
from .dataloader import ConvertToBraTSRegionsd

def get_test_loader(data_dir, json_list):
    data_dir = data_dir
    datalist_json = json_list
    test_files, _ = datafold_read(datalist=datalist_json, basedir=data_dir, fold=1, mode="testing")
    
    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            ConvertToBraTSRegionsd(keys=["label"]),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )

    test_ds = data.Dataset(data=test_files, transform=test_transform)
    test_loader = data.DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return test_loader