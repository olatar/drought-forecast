import os
import xarray as xr
import datetime

def save_xarray_as_netcdf(dataset, filename, folder_path='data/saves'):
    """
    Save an xarray dataset as a NetCDF file

    Parameters:
    dataset (xarray.Dataset): The dataset to save
    folder_path (str): The path to the folder where the NetCDF file will be saved
    filename (str): The name of the NetCDF file (default is '/data/saves')
    """
    today_yymmdd = datetime.datetime.now().strftime("%y%m%d")
    output_path = os.path.join(folder_path, filename + '_' + today_yymmdd + '.nc')
    dataset.to_netcdf(output_path)
    print(f"Saved dataset to {output_path}")


def load_netcdf_as_xarray(filename_wout_extension, most_recent=True, directory='data/saves'):
    """
    Load a NetCDF file as an xarray dataset

    Parameters:
    file_path (str): The path to the NetCDF file

    Returns:
    xarray.Dataset: The loaded dataset
    """
    if most_recent:
        latest_date = datetime.datetime.min
        latest_filename = None
        for file in os.listdir(directory):
            if file.endswith(".nc"):
                filename = file.split(".")[0][:-7]
                date = file.split(".")[0].split("_")[-1]
                date_formatted = datetime.datetime.strptime(date, "%y%m%d")
                
                if date_formatted > latest_date and filename_wout_extension == filename:
                    latest_date = date_formatted
                    latest_filename = file

        if latest_filename is not None:
            file_path = os.path.join(directory, latest_filename)
        else:
            raise FileNotFoundError("No NetCDF files found in the specified directory")
    else:
        file_path = os.path.join(directory, filename_wout_extension + '.nc')

    dataset_xa = xr.open_dataset(file_path)
    print(f"Loaded dataset from {file_path}")
    return dataset_xa