from osgeo import gdal
import numpy as np
from qgis.PyQt.QtWidgets import QAction, QFileDialog, QMessageBox
import os
from qgis.PyQt.QtWidgets import QAction, QFileDialog, QMessageBox
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
from matplotlib.backends.backend_pdf import PdfPages
from osgeo import osr
from scipy.spatial import distance



def read_gdal_array(filepath):
    ds = gdal.Open(filepath)
    array = ds.ReadAsArray()
    return array, ds

# Funzione per mappare i tipi GDAL → NumPy
def gdal_to_numpy_dtype(gdal_type):
    mapping = {
        gdal.GDT_Byte: np.uint8,
        gdal.GDT_UInt16: np.uint16,
        gdal.GDT_Int16: np.int16,
        gdal.GDT_UInt32: np.uint32,
        gdal.GDT_Int32: np.int32,
        gdal.GDT_Float32: np.float32,
        gdal.GDT_Float64: np.float64,
    }
    return mapping.get(gdal_type, np.float32)  # default fallback

def check_source(list_path):
            
            reference=list_path[0]
            # read the reference file
            reference_array, ds = read_gdal_array(reference)
            reference_shape = (ds.RasterYSize, ds.RasterXSize)
            geo_transform = ds.GetGeoTransform()
            projection = ds.GetProjection()
            datatype = ds.GetRasterBand(1).DataType
            np_dtype = gdal_to_numpy_dtype(datatype)
            
            for path in list_path:
                array, ds = read_gdal_array(path)
                # Check if the array shape matches the reference shape
                if array.shape != reference_shape or geo_transform != ds.GetGeoTransform() or projection != ds.GetProjection():
                    return False
            # If all checks pass,return True
            return True     

def save_band_to_file(output_path, band_data, dataset):
    """
    Save the band data (1 or more bands) to a GeoTIFF file.
    """
    driver = gdal.GetDriverByName('GTiff')

    # Detect number of bands
    if band_data.ndim == 2:
        band_data = np.expand_dims(band_data, axis=0)  # Make it (1, H, W)
    num_bands = band_data.shape[0]

    # print('num_bands:', num_bands)
    # print('band_data shape:', band_data.shape)
    # print('dataset shape:', dataset.RasterXSize, dataset.RasterYSize)

    output_dataset = driver.Create(output_path, dataset.RasterXSize, dataset.RasterYSize, num_bands, gdal.GDT_Float32)
    
    output_dataset.SetGeoTransform(dataset.GetGeoTransform())
    output_dataset.SetProjection(dataset.GetProjection())

    # Write each band
    for i in range(num_bands):
        output_dataset.GetRasterBand(i + 1).WriteArray(band_data[i])

    output_dataset.FlushCache()
    output_dataset = None

def generate_mask(input_path, mask_type, masking_value, output_path):
    
    # Open the input raster
    dataset = gdal.Open(input_path)
    if dataset is None:
        QMessageBox.critical(None, "Error", "Could not open input raster.")
        return
    
    if mask_type == 'discrete':
        # Create a mask for discrete values
        band = dataset.GetRasterBand(1)
        data = band.ReadAsArray()
        mask =  np.isin(data, masking_value, invert=True).astype(int)
    
    elif mask_type == 'continuous':
        # Create a mask for continuous values
        band = dataset.GetRasterBand(1)
        data = band.ReadAsArray()
        mask = np.where(data >= masking_value, 1, 0).astype(int)
        
    # save mask
    save_band_to_file(output_path, mask, dataset)
    
    return

def apply_mask(input_path, mask_path, output_path):
    
    
    flag = check_source([input_path, mask_path])
    
    if flag is False:
        return False
    
    
    #open the mask
    mask_dataset = gdal.Open(mask_path)
    # change 0 values to np.nan
    mask_band = mask_dataset.GetRasterBand(1)
    mask_data = mask_band.ReadAsArray()

    
    # open the input raster
    dataset = gdal.Open(input_path)
    

    
    # create the output raster
    driver = gdal.GetDriverByName('GTiff')
    output_dataset = driver.Create(output_path, dataset.RasterXSize, dataset.RasterYSize, dataset.RasterCount, gdal.GDT_Float32)
    # Set the geotransform and projection from the input dataset
    output_dataset.SetGeoTransform(dataset.GetGeoTransform())
    output_dataset.SetProjection(dataset.GetProjection())
    # Loop through each band in the input raster
    # and apply the mask
    # and save the masked data to the output raster at the end of the loop
    for i in range(1, dataset.RasterCount + 1):
        band = dataset.GetRasterBand(i)
        data = band.ReadAsArray().astype(np.float32)
        # Apply the mask
        data[mask_data == 0] = np.nan
        # Save the masked data to the output raster
        output_dataset.GetRasterBand(i).WriteArray(data)
        # Set the NoData value to NaN
        output_dataset.GetRasterBand(i).SetNoDataValue(np.nan)
    
    # Close the datasets
    dataset = None
    mask_dataset = None
    output_dataset.FlushCache()
    output_dataset = None  
    
    return
    
def generate_report(input_path, output_path, report_indices):

    # --- Load Raster ---
    ds = gdal.Open(input_path)
    array = ds.ReadAsArray()

    
    # Supponendo che `array` contenga tutte le bande del raster (es. array.shape = [num_bands, rows, cols])
    if array.ndim == 2:
        # Singola banda
        array = np.expand_dims(array, axis=0)
   
    
    num_bands = array.shape[0]
    boxplot_data = []
    outliers_info = []

    colors = [
        'blue', 'red', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta',
        'olive', 'gray', 'teal', 'pink', 'gold', 'navy', 'darkgreen', 'maroon',
        'deepskyblue', 'indigo', 'coral', 'lime'
    ]

    nodata_value = ds.GetRasterBand(1).GetNoDataValue()
    frequency_dict = None
    # --- Crea un PDF ---
    with PdfPages(output_path) as pdf:
        # --- Istogramma combinato ---
        if num_bands > 1:
            if 1 in report_indices:
                fig1=plt.figure(figsize=(10, 5))
                for i in range(num_bands):
                    data = array[i]
                    exclude_nodata = data[data != nodata_value].ravel()                        
                    counts, bin_edges = np.histogram(exclude_nodata, bins=100)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    color = colors[i % len(colors)]
                    plt.plot(bin_centers, counts, label=f'Banda {i+1}', color=color)
                        
                plt.title('Raster Distribution')
                plt.xlabel('Pixel Value')
                plt.ylabel('Absolute Frequency')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                pdf.savefig(fig1)
                plt.close()

                # --- Subplot per ogni banda ---
                fig, axes = plt.subplots(nrows=num_bands, figsize=(10, 3 * num_bands))
                for i in range(num_bands):
                    data = array[i]
                    exclude_nodata = data[data != nodata_value].ravel()
                    counts, bin_edges = np.histogram(exclude_nodata, bins=100)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    axes[i].plot(bin_centers, counts, color='blue')
                    axes[i].set_title(f'Band {i+1} Distribution')
                    axes[i].set_xlabel('Pixel Value')
                    axes[i].set_ylabel('Absolute Frequency')
                    axes[i].grid(True)
                
                    if 2 in report_indices:
                
                        boxplot_data.append(exclude_nodata)
                        q1 = np.percentile(exclude_nodata, 25)
                        q3 = np.percentile(exclude_nodata, 75)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        outliers = exclude_nodata[(exclude_nodata < lower_bound) | (exclude_nodata > upper_bound)]
                        outliers_count = len(outliers)
                        outliers_percentage = (outliers_count / len(exclude_nodata)) * 100

                        if outliers_count > 0:
                            outliers_info.append((f'Band {i+1}', 'yes', outliers_count, outliers_percentage))
                        else:
                            outliers_info.append((f'Band {i+1}', 'no'))
                
                
            plt.tight_layout()
            pdf.savefig()
            plt.close()
                    

                        
                            
        elif num_bands == 1:
            data = array[0]
            exclude_nodata = data[data != nodata_value].ravel()
            
            # Singola banda
            if 1 in report_indices:
                plt.figure(figsize=(10, 5))
                
                counts, bin_edges = np.histogram(exclude_nodata, bins=100)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                plt.plot(bin_centers, counts, color='blue')
                plt.title('Raster Distribution - Single Band')
                plt.xlabel('Pixel Value')
                plt.ylabel('Absolute Frequency')
                plt.grid(True)
                plt.tight_layout()
                pdf.savefig()
                plt.close()
                
            if 2 in report_indices:
                boxplot_data.append(exclude_nodata)
                q1 = np.percentile(exclude_nodata, 25)
                q3 = np.percentile(exclude_nodata, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = exclude_nodata[(exclude_nodata < lower_bound) | (exclude_nodata > upper_bound)]
                outliers_count = len(outliers)
                outliers_percentage = (outliers_count / len(exclude_nodata)) * 100

                if outliers_count > 0:
                    outliers_info.append(('Band 1', 'yes', outliers_count, outliers_percentage))
                else:
                    outliers_info.append(('Band 1', 'no'))
                
                
                # for each single value present in the array count the frequency
            unique, unique_count = np.unique(exclude_nodata, return_counts=True)
            # if the number of unique values is greater than 20 generete an error
            if len(unique) > 20:
                QMessageBox.warning(None, "Error", "Too many unique values")
                frequency_dict = None
            else:
                # create a dictionary with the unique values and their frequency
                frequency_dict = dict(zip(unique, unique_count))
           
           

        # --- Boxplot ---
        if 2 in report_indices:
            plt.figure(figsize=(12, 6))
            box = plt.boxplot(boxplot_data, vert=True, patch_artist=True, showfliers=False)
                    
            plt.title('Raster Boxplot')
            plt.xlabel('Bands')
            plt.ylabel('Pixel Value')
            plt.xticks(ticks=range(1, num_bands + 1), labels=[f'Band {i+1}' for i in range(num_bands)])
            plt.grid(True)
            plt.tight_layout()
                       
            pdf.savefig()
            plt.close()

        # --- Pagina testuale con info statistiche ---
        
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        text_lines = []


            
        if 3 in report_indices:
            
            # --- Informazioni geografiche ---
            geo_transform = ds.GetGeoTransform()
            crs= ds.GetProjectionRef()
            # take the short name of the crs
            
            # Ottieni il sistema di riferimento spaziale dal dataset
            proj_wkt = ds.GetProjection()
            srs = osr.SpatialReference()
            srs.ImportFromWkt(proj_wkt)

            # Prova a ottenere il codice EPSG
            epsg_code = srs.GetAttrValue("AUTHORITY", 1)
                
            resolution = geo_transform[1]
            
            text_lines.append("\nGeographic Information:\n")
            text_lines.append(f"CRS: EPSG {epsg_code}")
            text_lines.append(f"\nResolution: {resolution} m")
            
            # --- Informazioni statistiche ---            
            text_lines.append("\nStatistical Information:\n")
            for i in range(num_bands):
                data = array[i]
                exclude_nodata = data[data != nodata_value].ravel()
                mean = np.mean(exclude_nodata)
                variance = np.var(exclude_nodata)
                median = np.median(exclude_nodata)
                
                min_val = np.min(exclude_nodata)
                max_val = np.max(exclude_nodata)
                
                text_lines.append(f"Band {i+1}: Mean: {mean:.2f}, Variance: {variance:.2f}, Median: {median:.2f}")
                text_lines.append(f"        Min: {min_val:.2f}, Max: {max_val:.2f}")

            
            
            if frequency_dict is not None:
                text_lines.append("\nFrequency Dictionary:\n")
                for key, value in frequency_dict.items():
                    text_lines.append(f"Value {key}: {value}")
                    
                text_lines.append("\nClass Area (m²):\n")
                for key, value in frequency_dict.items():
                    area = value * resolution * resolution
                    text_lines.append(f"Class {key}: {area:.2f} m²")

            
        if 2 in report_indices:
            text_lines.append("\nOutliers Information:\n")
            for info in outliers_info:
                if len(info) == 4:
                    text_lines.append(f"{info[0]}: Outliers present - Count: {info[2]}, Percentage: {info[3]:.2f}%")
                else:
                    text_lines.append(f"{info[0]}: No outliers present")
        
        # --- Impaginazione su più pagine nel PDF ---
        max_lines_per_page = 30
        for i in range(0, len(text_lines), max_lines_per_page):
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')

            chunk = text_lines[i:i + max_lines_per_page]
            page_text = "\n".join(chunk)

            ax.text(0, 1, page_text, va='top', ha='left', fontsize=12, wrap=True, family='monospace')
            pdf.savefig()
            plt.close()

    # print(f"PDF report salvato in: {output_path}")
    return
    
def generate_comb_report(layer_continous, layer_discrete, output_path, report_indices):
        # --- Load Raster ---
    ds_continous = gdal.Open(layer_continous)
    array_continous = ds_continous.ReadAsArray()
    
    nodata_value_continous = ds_continous.GetRasterBand(1).GetNoDataValue()
    # remove nodata values
    array_continous = array_continous[array_continous != nodata_value_continous]
    
    
    ds_discrete = gdal.Open(layer_discrete)
    array_discrete = ds_discrete.ReadAsArray()
    
    nodata_value_discrete = ds_discrete.GetRasterBand(1).GetNoDataValue()
    # remove nodata values
    array_discrete = array_discrete[array_discrete != nodata_value_discrete]
        
    # check if the two arrays have the same shape dimension and resolution
    flag=check_source([layer_continous, layer_discrete])
    
    if flag is False:
        return False
    
    #take unique values from the discrete array
    unique_values = np.unique(array_discrete)
    
    # # stack the two arrays
    # stacked_array = np.dstack((array_continous, array_discrete))
    
    # for each unique value found in the stacked array and take the corresponding values from the continous array
    # and save them in a dictionary
   # Dizionario per salvare le statistiche
    stats_dict = {}

    for value in unique_values:
        mask = array_discrete == value
        values = array_continous[mask]

        # Escludi eventuali valori NaN o nodata se necessario
        values = values[~np.isnan(values)]

        if values.size > 0:
            stats = {
                "count": len(values),
                "mean": np.mean(values),
                "var": np.var(values),
                "min": np.min(values),
                "max": np.max(values),
                "median": np.median(values),
            }
        else:
            stats = {
                "count": 0,
                "mean": None,
                "var": None,
                "min": None,
                "max": None,
                "median": None,
            }

        stats_dict[value] = stats
        
    # --- Crea un PDF ---
    with PdfPages(output_path) as pdf:
        # --- Istogramma per ciascun unique value colonna---
        if 1 in report_indices:
            
            fig, axes = plt.subplots(nrows=len(unique_values), figsize=(10, 3 * len(unique_values)))
            if len(unique_values) == 1:
                axes = [axes]  # Assicura che axes sia sempre iterabile

            for i, value in enumerate(unique_values):
                mask = array_discrete == value
                values = array_continous[mask]

                # Escludi eventuali valori NaN o nodata se necessario
                values = values[~np.isnan(values)]

                counts, bin_edges = np.histogram(values, bins=100)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                axes[i].bar(bin_centers, counts, width=bin_edges[1] - bin_edges[0], color='skyblue', edgecolor='black')
                axes[i].set_title(f'Class {int(value)} Distribution')
                axes[i].set_xlabel('Pixel Value')
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True)

            plt.tight_layout()
            pdf.savefig()
            plt.close()

        if 2 in report_indices:
            outliers_info = []
            # --- one Boxplot with for each unique value  ---
            fig, ax = plt.subplots(figsize=(12, 6))
            box = ax.boxplot([array_continous[array_discrete == value] for value in unique_values], vert=True, patch_artist=True, showfliers=False)
            ax.set_title('Raster Boxplot')
            ax.set_xlabel('Classes')
            ax.set_ylabel('Pixel Value')
            ax.set_xticks(range(1, len(unique_values) + 1))
            ax.set_xticklabels([f'Class {int(value)}' for value in unique_values])
            ax.grid(True)
            plt.tight_layout()
            pdf.savefig()
            plt.close()
       
            # for each boxplot verify the presence of outliers
            for i, value in enumerate(unique_values):
                mask = array_discrete == value
                values = array_continous[mask]

                # Escludi eventuali valori NaN o nodata se necessario
                values = values[~np.isnan(values)]

                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = values[(values < lower_bound) | (values > upper_bound)]
                outliers_count = len(outliers)
                outliers_percentage = (outliers_count / len(values)) * 100

                if outliers_count > 0:
                    outliers_info.append((f"Class {int(value)}",'yes', outliers_count, outliers_percentage))
                else:
                    outliers_info.append((f"Class {int(value)}",'no'))        
            
            
        
        # --- Pagina testuale con info statistiche ---
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        text_lines = []

        # --- Informazioni geografiche ---
        geo_transform = ds_continous.GetGeoTransform()
        crs= ds_continous.GetProjectionRef()
        # take the short name of the crs
        
        # Ottieni il sistema di riferimento spaziale dal dataset
        proj_wkt = ds_continous.GetProjection()
        srs = osr.SpatialReference()
        srs.ImportFromWkt(proj_wkt)

        # Prova a ottenere il codice EPSG
        epsg_code = srs.GetAttrValue("AUTHORITY", 1)
            
        resolution = geo_transform[1]
        
        text_lines.append("\nGeographic Information:\n")
        text_lines.append(f"CRS: EPSG {epsg_code}")
        text_lines.append(f"\nResolution: {resolution} m")
        
        if 3 in report_indices:
            # --- Informazioni statistiche ---
            text_lines.append("\nStatistical Information:\n")
            for value, stats in stats_dict.items():
                text_lines.append(f"Class {value}: Mean: {stats['mean']:.2f}, Variance: {stats['var']:.2f}, Median: {stats['median']:.2f}")
                text_lines.append(f"         Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")
                text_lines.append(f"         Count: {stats['count']}, Area: {stats['count'] * resolution * resolution:.2f} m²")
            
        if 2 in report_indices:
            # --- Informazioni sugli outliers ---
            text_lines.append("\nOutliers Information:\n")
            for info in outliers_info:
                if len(info) == 4:
                    text_lines.append(f"{info[0]}: Outliers present - Count: {info[2]}, Percentage: {info[3]:.2f}%")
                else:
                    text_lines.append(f"{info[0]}: No outliers present")
        
        
        # --- Impaginazione su più pagine nel PDF ---
        max_lines_per_page = 30
        for i in range(0, len(text_lines), max_lines_per_page):
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')

            chunk = text_lines[i:i + max_lines_per_page]
            page_text = "\n".join(chunk)

            ax.text(0, 1, page_text, va='top', ha='left', fontsize=12, wrap=True, family='monospace')
            pdf.savefig()
            plt.close()   
    # print(f"PDF report salvato in: {output_path}")

    return

def compute_change_detection(input_path_t0, input_path_t1, output_path):
    
    flag=check_source([input_path_t0, input_path_t1])
    print('flag:', flag)
    if flag is False:
        return False
    
    # Open the input rasters
    dataset_t0 = gdal.Open(input_path_t0)
    dataset_t1 = gdal.Open(input_path_t1)
    array_t0 = dataset_t0.ReadAsArray().astype(np.float32)
    array_t1 = dataset_t1.ReadAsArray().astype(np.float32)
    
       
    # if the number of bands is different between the two rasters generate an error
    if dataset_t0.RasterCount != dataset_t1.RasterCount:
        QMessageBox.critical(None, "Error", "The number of bands is different between the two rasters.")
        return
    
    # if the dataset has 1 band expand the dimension
    if array_t0.ndim == 2:
        array_t0 = np.expand_dims(array_t0, axis=0)
        array_t1 = np.expand_dims(array_t1, axis=0)
    
    methods = [item[1] for item in output_path]
    difference = array_t1 - array_t0
    # print('difference shape:', difference.shape)
    
    if 'difference' in methods:
        # Calculate the difference
        out_path= [item[0] for item in output_path if item[1] == 'difference'][0]
        # print('diff shape:', difference.shape)
        #save difference to file USING GDAL
        save_band_to_file(out_path, difference, dataset_t1)
        
    if 'euclidean' in methods:
        # Calculate the Euclidean distance using scipy
        distance_map = np.linalg.norm(difference, axis=0)
        
        # print('distance_map shape:', distance_map.shape)
        out_path= [item[0] for item in output_path if item[1] == 'euclidean'][0]
        save_band_to_file(out_path, distance_map, dataset_t1)
    
    if 'hamming' in methods:
        # if and only if the two rasters have 1 band and the values are 0 and 1
        if dataset_t0.RasterCount != 1 or dataset_t1.RasterCount != 1:
            QMessageBox.critical(None, "Error", "The rasters must have only one band and the values must be 0 and 1.")
            return
        if np.max(array_t0) > 1 or np.max(array_t1) > 1:
            QMessageBox.critical(None, "Error", "The rasters must have only one band and the values must be 0 and 1.")
            return
        
        # Calculate the Hamming distance
       
        frequency_dict = {}
        
        if 'Hamm_report' in methods:
            # take unique values from the diff array
            unique_values = np.unique(difference)
            # create a dictionary with the unique values and their frequency
            unique, unique_count = np.unique(difference, return_counts=True)
            # save the unique values and their frequency in a dictionary
            frequency_dict = dict(zip(unique, unique_count))
            # calculate the total number of pixels
            tot_count = np.sum(unique_count)
        
            out_path_report= [item[0] for item in output_path if item[1] == 'Hamm_report'][0]
            # save the dictionary to a pdf 
            with PdfPages(out_path_report) as pdf:
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis('off')
                text_lines = []
                text_lines.append("\nReport Hamming Distance\n")
                    
                # --- Informazioni geografiche ---
                geo_transform = dataset_t0.GetGeoTransform()
                crs= dataset_t0.GetProjectionRef()
                # take the short name of the crs
                
                # Ottieni il sistema di riferimento spaziale dal dataset
                proj_wkt = dataset_t0.GetProjection()
                srs = osr.SpatialReference()
                srs.ImportFromWkt(proj_wkt)

                # Prova a ottenere il codice EPSG
                epsg_code = srs.GetAttrValue("AUTHORITY", 1)
                    
                resolution = geo_transform[1]
                
                text_lines.append("\nGeographic Information:\n")
                text_lines.append(f"CRS: EPSG {epsg_code}")
                text_lines.append(f"\nResolution: {resolution} m")               
                
                text_lines.append("\nDistance:\n")
                
                # hamming distance
                
                no_zero_count = np.sum(unique_count[unique != 0])
                # calculate the percentage of the hamming distance
                hamming_distance = no_zero_count/tot_count * 100
                text_lines.append(f"Hamming Distance: {hamming_distance:.2f} %")
                
                for key, value in frequency_dict.items():
                    # compute the percentage of the value
                    percentage = value / tot_count * 100
                    # calculate the area in m²
                    area = value * resolution * resolution
                    text_lines.append(f"Value {key}: {value}, Percentage: {percentage:.2f}, Area: {area:.2f} m²")
                            
                # --- Impaginazione su più pagine nel PDF ---
                max_lines_per_page = 30
                for i in range(0, len(text_lines), max_lines_per_page):
                    fig, ax = plt.subplots(figsize=(8.5, 11))
                    ax.axis('off')

                    chunk = text_lines[i:i + max_lines_per_page]
                    page_text = "\n".join(chunk)

                    ax.text(0, 1, page_text, va='top', ha='left', fontsize=12, wrap=True, family='monospace')
                    pdf.savefig()
                    plt.close()

        
        out_path= [item[0] for item in output_path if item[1] == 'hamming'][0]
        # print(out_path)
        save_band_to_file(out_path, difference, dataset_t1)
    
    return

def validate_discretization(values):
    # 1. Verifica che i primi elementi (classi) siano unici
    class_ids = [cls for cls, _, _ in values]
    if len(class_ids) != len(set(class_ids)):
        return 1 # Classi duplicate

    # 2. Verifica che gli intervalli non si sovrappongano
    # Ordina per lower bound
    sorted_intervals = sorted(values, key=lambda x: x[1])

    for i in range(len(sorted_intervals) - 1):
        _, _, upper1 = sorted_intervals[i]
        _, lower2, _ = sorted_intervals[i + 1]
        if upper1 > lower2:
            return 2 # Intervalli sovrapposti

    return 0  # Tutto ok

def classify(input_path, table_values, output_path):
    
    # open the input raster
    dataset = gdal.Open(input_path)
    if dataset is None:
        return  3
    raster= dataset.ReadAsArray()
    
    # check if the number of bands is 1
    if raster.ndim == 3:
        return 4
    
    discrete_raster = np.zeros_like(raster, dtype=np.uint8) # Set NoData values to class 0
    discrete_raster=discrete_raster + 99  # Initialize with NoData value (class 99)
    # Initialize with zeros (class 0)
    for class_id, lower_bound, upper_bound in table_values:
        mask = (raster > lower_bound) & (raster <= upper_bound)
        discrete_raster[mask] = class_id
    
    # save the classified raster
    save_band_to_file(output_path, discrete_raster, dataset)
    #close the dataset
    dataset = None

    return 0 # Successo
