import os
import xarray as xr
from utils import load_netcdf_as_xarray, save_xarray_as_netcdf
import datetime
import calendar
from scipy import stats
import numpy as np

MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def load_ecmwf_netcdfs_to_xarray(directory):

    nc_files = [
        os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".nc")
    ]
    nc_files_sorted = sorted(
        nc_files, key=lambda x: int(x.split("_")[-1].split(".")[0][2:])
    )

    datasets_dict = {}
    for file in nc_files_sorted:
        # Assuming that 'imX' in filename corresponds to month number X
        month_num = int(
            file.split("_")[-1].split(".")[0][2:]
        )  # Extract the month number
        month_name = MONTHS[
            month_num - 1
        ]  # Get the corresponding month name from the list
        dataset = xr.open_dataset(file)
        datasets_dict[month_name] = (
            dataset  # Use the month name as the key in the dictionary
        )
        print(f"Loaded file for {month_name}: {file}")

    datasets_list = [datasets_dict[month] for month in MONTHS]

    # Rename 'month' variable if it exists to avoid conflict
    for i, (month, dataset) in enumerate(zip(MONTHS, datasets_list)):
        dataset = dataset.expand_dims("month").assign_coords(month=[month])
        datasets_list[i] = dataset

    dataset_xa = xr.concat(datasets_list, dim="month")

    # -------
    # Select the subset for January and drop the 'month' dimension
    january_subset = dataset_xa.sel(month="March").drop_vars("month")

    # Compare the datasets
    are_identical = datasets_dict["March"].equals(january_subset)

    # Print the result
    print(f"Are the datasets identical? {are_identical}")
    # -------

    return dataset_xa


def save_xarray_as_netcdf(dataset, filename, folder_path="data/saves"):
    """
    Save an xarray dataset as a NetCDF file

    Parameters:
    dataset (xarray.Dataset): The dataset to save
    folder_path (str): The path to the folder where the NetCDF file will be saved
    filename (str): The name of the NetCDF file (default is '/data/saves')
    """
    if filename.endswith(".nc"):
        filename = filename[:-3]

    today_yymmdd = datetime.datetime.now().strftime("%y%m%d")
    output_path = os.path.join(folder_path, filename + "_" + today_yymmdd + ".nc")

    dataset.to_netcdf(output_path)
    print(f"Saved dataset to {output_path}")


def load_netcdf_as_xarray(filename, most_recent=True, directory="data/saves"):
    """
    Load a NetCDF file as an xarray dataset

    Parameters:
    file_path (str): The path to the NetCDF file

    Returns:
    xarray.Dataset: The loaded dataset
    """
    if filename.endswith(".nc"):
        filename = filename[:-3]

    if most_recent:
        latest_date = datetime.datetime.min
        latest_filename = None
        for file in os.listdir(directory):
            if file.endswith(".nc"):
                filename_temp = file.split(".")[0][:-7]
                date = file.split(".")[0].split("_")[-1]
                date_formatted = datetime.datetime.strptime(date, "%y%m%d")

                if date_formatted > latest_date and filename == filename_temp:
                    latest_date = date_formatted
                    latest_filename = file

        if latest_filename is not None:
            file_path = os.path.join(directory, latest_filename)
        else:
            raise FileNotFoundError("No NetCDF files found in the specified directory")
    else:
        file_path = os.path.join(directory, filename + ".nc")

    dataset_xa = xr.open_dataset(file_path)
    print(f"Loaded dataset from {file_path}")

    return dataset_xa


def rename_netcdf_dimension(dataset, dict, save=False, save_filename=None):
    dataset = dataset.rename(dict)
    if save:
        save_xarray_as_netcdf(dataset, save_filename)
    return dataset


# See which years missing in xarray


def find_missing_years(datasets_dict, possible_years=range(1993, 2024)):

    months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    for month_name in months:
        # Get the available years for the current month
        available_years = set(datasets_dict[month_name].year.values)

        # Calculate the missing years
        missing_years = possible_years - available_years

        # Print the missing years for the current month
        print(f"Missing years for {month_name}: {sorted(missing_years)}")


def remove_year(dataset, year, save=False, save_filename=None):
    dataset = dataset.sel(year=~dataset.year.isin([year]))
    if save:
        save_xarray_as_netcdf(dataset, save_filename)
    return dataset


# dataset_xa.sel(ensemble_member=slice(0, 24))
def select_ensemble_members(dataset, start=0, end=24, save=False, save_filename=None):
    dataset = dataset.sel(ensemble_member=slice(start, end))
    if save:
        save_xarray_as_netcdf(dataset, save_filename)
    return dataset


# test the functions
# directory = 'data/ecmwf'
# dataset = load_ecmwf_netcdfs_to_xarray(directory)
# save_xarray_as_netcdf(dataset, filename="ecmwf_raw")


# get number of days given a year and month
def days_per_year_month(year: int, month: str):
    month: int = list(calendar.month_abbr).index(month[:3].capitalize())
    return calendar.monthrange(year, month)[1]


def month_idx_as_string(reference_month, month_number):
    # Get the index of the reference month in the calendar
    reference_month_index = list(calendar.month_abbr).index(
        reference_month[:3].capitalize()
    )

    # Calculate the index of the target month
    target_month_index = (reference_month_index + month_number - 2) % 12

    # Get the month string
    month_string = calendar.month_name[target_month_index + 1]
    return month_string


def get_number_of_days(reference_year, reference_month, time_index):
    # Get the month string based on the reference month and time index
    forecast_month = month_idx_as_string(reference_month, time_index)

    # Check if the forecasted month belongs to the next year
    if time_index > 1 and forecast_month == month_idx_as_string(reference_month, 1):
        next_year = reference_year + 1
    else:
        next_year = reference_year

    # Calculate the number of days for the forecasted month
    number_of_days = days_per_year_month(next_year, forecast_month)

    return number_of_days


# def per_second_to_per_month(
# write def per_second_to_per_month code


def modify_per_second_to_per_month(
    dataset_xa, seconds_in_day=86400, save=False, save_filename=None
):
    for year in dataset_xa.year.values:
        for month in dataset_xa.month.values:
            for forecast_index in dataset_xa.forecast_index.values:

                # Calculate the number of days for the respective month and year
                number_of_days = get_number_of_days(year, month, forecast_index)
                conversion_factor = seconds_in_day * number_of_days

                # Select the data for this specific year, month, and forecast index
                data_subset = dataset_xa.sel(
                    year=year, month=month, forecast_index=forecast_index
                )

                # Multiply the data by the conversion factor
                # Note: we need to broadcast the conversion factor to match the data dimensions if it isn't already
                data_subset["tprate"] *= conversion_factor

                # Assign the adjusted values back to the dataset
                dataset_xa.loc[
                    {"year": year, "month": month, "forecast_index": forecast_index}
                ] = data_subset["tprate"]

    if save:
        save_xarray_as_netcdf(dataset_xa, save_filename)

    return dataset_xa


def sum_across_lat_lon(dataset_xa, save=False, save_filename=None):
    dataset_xa = dataset_xa.sum(dim=["longitude", "latitude"])
    if save:
        save_xarray_as_netcdf(dataset_xa, save_filename)
    return dataset_xa


def aggregate_across_ensemble_members(
    dataset_xa, method, save=False, save_filename=None
):
    if method == "mean":
        dataset_xa = dataset_xa.mean(dim=["ensemble_member"])
    elif method == "median":
        dataset_xa = dataset_xa.median(dim=["ensemble_member"])
    else:
        raise ValueError(
            "Invalid aggregation method. Choose between 'mean' and 'median'."
        )

    if save:
        save_xarray_as_netcdf(dataset_xa, save_filename)
    return dataset_xa


def spi_of_timeseries(precip_data_each_year):

    # Fit gamma distribution to positive values (floc forces loc to 0, thus output loc will be 0)
    shape, loc, scale = stats.gamma.fit(precip_data_each_year, floc=0)

    # GAMMA FIT & CDF VALUE: gamma_cdf = stats.gamma.cdf(adjusted_precip_data, a=shape, loc=loc, scale=scale)
    gamma_cdf = stats.gamma.cdf(precip_data_each_year, a=shape, scale=scale, loc=loc)

    # NORMAL DIst: Convert the gamma CDF to SPI values
    spi_values = stats.norm.ppf(gamma_cdf)

    return spi_values, gamma_cdf, shape, loc, scale


def create_spi_data_structure(dataset_xa):
    # Create a new dataset with the desired dimensions and coordinates
    xarray = xr.DataArray(
        data=np.nan,  # Initialize with NaNs
        dims=["year", "month", "forecast_index", "ensemble_member"],
        coords={
            "year": dataset_xa.year.values,
            "month": dataset_xa.month.values,
            "forecast_index": dataset_xa.forecast_index.values,
            "ensemble_member": dataset_xa.ensemble_member.values,
        },
        name="SPI-1",
    )

    spi_1_per_ensemble_member = xr.Dataset({"spi_1_values": xarray})
    return spi_1_per_ensemble_member




'''
# Create a new dataset with the desired dimensions and coordinates
xarray = xr.DataArray(
    data=np.nan,  # Initialize with NaNs
    dims=['year', 'month', 'forecast_index', 'ensemble_member'],
    coords={
        'year': dataset_xa.year.values,
        'month': dataset_xa.month.values,
        'forecast_index': dataset_xa.forecast_index.values,
        'ensemble_member': dataset_xa.ensemble_member.values
    },
    name='SPI-1'
)

spi_1_per_ensemble_member = xr.Dataset({'SPI-1': xarray})
spi_1_per_ensemble_member

'''


"""
            # Fit gamma distribution to positive values (floc forces loc to 0, thus output loc will be 0)
            shape, loc, scale = stats.gamma.fit(precip_data_each_year, floc=0)

            # GAMMA FIT & CDF VALUE: gamma_cdf = stats.gamma.cdf(adjusted_precip_data, a=shape, loc=loc, scale=scale)
            gamma_cdf = stats.gamma.cdf(precip_data_each_year, a=shape, scale=scale, loc=loc)

            # NORMAL DIst: Convert the gamma CDF to SPI values
            spi_values = stats.norm.ppf(gamma_cdf)
"""


dataset_xa = load_netcdf_as_xarray("ecmwf_unit_change")

# rename_netcdf_dimension(dataset_xa, dict={'time': 'forecast_index', 'number': 'ensemble_member'}, save=True, save_filename="ecmwf_renamed")
# remove_year(dataset_xa, year=2022, save=True, save_filename="ecmwf_wout_2022")
# modify_per_second_to_per_month(dataset_xa, start=0, end=24, save=True, save_filename="ecmwf_unit_change")

# modify_per_second_to_per_month(dataset_xa, save=True, save_filename="ecmwf_unit_change")

# sum_across_lat_lon(dataset_xa, save=True, save_filename="ecmwf_sum_lat_lon")

# aggregate_across_ensemble_members(dataset_xa, method='mean', save=True, save_filename="ecmwf_ensemble__mean")
# aggregate_across_ensemble_members(dataset_xa, method='median', save=True, save_filename="ecmwf_ensemble__median")


print(create_spi_data_structure(dataset_xa))