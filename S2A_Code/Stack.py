from osgeo import gdal
import numpy as np
from qgis.PyQt.QtWidgets import QAction, QFileDialog, QMessageBox
import os


native_res = {
    10: ["2", "3", "4", "8"],
    20: ["1", "2", "3", "4", "5", "6", "7", "8A", "11", "12", "SCL"],
    60: ["1", "2", "3", "4", "5", "6", "7", "8A", "9", "10", "11", "12", "SCL"]
}

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
            
             # Create an empty array to hold the stacked data
            stacked_array = np.zeros((len(list_path), reference_shape[0], reference_shape[1]), dtype=np_dtype)
            
            i= 0
            
            for path in list_path:
                array, ds = read_gdal_array(path)
                # Check if the array shape matches the reference shape
                if array.shape != reference_shape or geo_transform != ds.GetGeoTransform() or projection != ds.GetProjection():
                    flag= False
                    return None, None, None, None, None, flag

            else:
                # Stack the array
                stacked_array[i] = array
                i += 1
                flag = True
            
            return reference_shape, geo_transform, projection, stacked_array, datatype,flag
        

def resample_with_gdal(input_path, out_xsize, out_ysize, method=gdal.GRA_NearestNeighbour):
    tmp_path = '/vsimem/temp_resample.tif'
    try:
        gdal.Warp(tmp_path, input_path, width=out_xsize, height=out_ysize, resampleAlg=method)
        out_ds = gdal.Open(tmp_path)
        if out_ds is None:
            raise ValueError(f"Failed to open resampled file from {input_path}")
            
        array = out_ds.ReadAsArray()
        if array is None:
            raise ValueError(f"Failed to read resampled array from {input_path}")
            
        return array
    except Exception as e:
        raise ValueError(f"Resampling error: {str(e)}")
    finally:
        gdal.Unlink(tmp_path)
        
        

def found_folder(path_folder, resolution):
    resolution = f"R{resolution}m"
    for root, dirs, _ in os.walk(path_folder):
        for dir_name in dirs:
            if resolution in dir_name:
                return os.path.join(root, dir_name)
    return None




def stack_generator(input_path, output_path, scenario, resolution, band_list ):

    """ 
    input_path: path to the input file
    output_path: path to the output file
    scenario: (not yet fully implemented)
    resolution: integer
    band_list: list of (origine, destinazione)
    """
    # scenario 1
    
    if scenario == 1:
        
        # print("Stacking bands from input file:", input_path)
        # read the input file
        dataset = gdal.Open(input_path)
        
        # if the number of bands is less than the number of bands in the list, raise an error
        num_bands = dataset.RasterCount
        if num_bands < len(band_list):
            return 2
        
        
        if dataset is None:
            raise RuntimeError(f"Cannot open file: {input_path}")

        # Get the geotransform and projection
        gt = dataset.GetGeoTransform()
        proj = dataset.GetProjection()

        band_dict = {}
        nodata_dict = {}

        # Loop through the band list and read the bands
        for origine, destinazione, name,_ in band_list:
            band = dataset.GetRasterBand(origine)
            if band is None:
                return 3
            
            array = band.ReadAsArray()
            nodata = band.GetNoDataValue()
            #band_dict[destinazione] = array
            nodata_dict[destinazione] = nodata
            # modifica del 1/07/2025 aggiungere il nome della banda al dizionario
            band_dict[destinazione] = (array, name)
        
        # Check if the bands are in the correct order
        sorted_positions = sorted(band_dict.keys())
        sorted_bands = [band_dict[pos] for pos in sorted_positions]
        sorted_nodata = [nodata_dict[pos] for pos in sorted_positions]

        sorted_arrays = [band_data[0] for band_data in sorted_bands]  # Solo gli array
        sorted_names = [band_data[1] for band_data in sorted_bands]   # Solo i nomi
        
        stacked = np.dstack(sorted_arrays)

        # Create the output dataset      
        driver = gdal.GetDriverByName('GTiff')
        rows, cols = stacked.shape[0], stacked.shape[1]
        out_dataset = driver.Create(output_path, cols, rows, len(sorted_bands), gdal.GDT_Float32)
        out_dataset.SetGeoTransform(gt)
        out_dataset.SetProjection(proj)

        # write the stacked bands to the output dataset
        for i, (band_array,band_name, nodata_val) in enumerate(zip(sorted_arrays,sorted_names, sorted_nodata)):
            out_band = out_dataset.GetRasterBand(i + 1)
            out_band.WriteArray(band_array)
            
            # modifica del 1/07/2025 aggiungere il nome della banda al dizionario
            # set the name of the band
            out_band.SetDescription(band_name)
            
            if nodata_val is not None:
                out_band.SetNoDataValue(nodata_val)

        # close the datasets
        dataset = None
        out_dataset = None
        
        # print("Stacked and saved to:", output_path)

        return True



    # scenario 2    
    if scenario == 2:        
        reference_shape = None
        geo_transform = None
        projection = None
        
        # Find folder with requested resolution
        folder_path = found_folder(input_path, resolution)
        if not folder_path:
            print(f"No folder found with resolution {resolution}m")
            return False

        # Get list of available files
        files = [f for f in os.listdir(folder_path) if f.endswith('.jp2') or f.endswith('.tif')]
        
        # Find reference dimensions from any valid band
        for band_id, _, _,_ in band_list:
            # if band_id is 9 we need to check for 8A
            if band_id == 9:
                band_str = "8A"
            # as a result we need to check for band_id minus 1 from band id greater than 9
            elif band_id > 9:
                # Check if band_id is 13 (SCL) and adjust accordingly
                if band_id == 14:
                    band_str = "SCL"
                else:
                    band_str = str(band_id-1).zfill(2)
            else:
                band_str = str(band_id).zfill(2)
            
            #print(f"LEGGGIMI band_str is: {band_str}")
            # Check both possible file naming patterns
            for name_pattern in [f"B{band_str}_{resolution}m", f"B{band_str}_{resolution}m.jp2", f"{band_str}_{resolution}m.jp2"]:
                matching_files = [f for f in files if name_pattern in f]
                if matching_files:
                    ref_path = os.path.join(folder_path, matching_files[0])
                    try:
                        array, ds = read_gdal_array(ref_path)
                        reference_shape = (ds.RasterYSize, ds.RasterXSize)
                        geo_transform = ds.GetGeoTransform()
                        projection = ds.GetProjection()
                        print(f"Found reference dimensions: {reference_shape} from {matching_files[0]}")
                        break
                    except Exception as e:
                        print(f"Error reading reference file {ref_path}: {e}")
            if reference_shape:
                break

        # If no reference found at requested resolution, try other resolutions
        if reference_shape is None:
            for r in [10, 20, 60]:
                if r == resolution:
                    continue
                temp_folder = found_folder(input_path, r)
                if temp_folder:
                    temp_files = [f for f in os.listdir(temp_folder) if f.endswith('.jp2') or f.endswith('.tif')]
                    if temp_files:
                        ref_path = os.path.join(temp_folder, temp_files[0])
                        try:
                            array, ds = read_gdal_array(ref_path)
                            # CORREZIONE: Il fattore deve essere calcolato correttamente
                            # r = risoluzione dei dati disponibili, resolution = risoluzione target desiderata
                            factor = r / resolution  # Se r=20m e resolution=10m, factor=2 (raddoppiare le dimensioni)
                            reference_shape = (int(ds.RasterYSize * factor), int(ds.RasterXSize * factor))
                            # Adjust geotransform for new resolution (pixel size target)
                            original_gt = ds.GetGeoTransform()
                            # Il pixel size finale deve essere uguale alla resolution target
                            target_pixel_size_x = resolution if original_gt[1] > 0 else -resolution
                            target_pixel_size_y = -resolution if original_gt[5] < 0 else resolution
                            geo_transform = (
                                original_gt[0], 
                                target_pixel_size_x,
                                original_gt[2],
                                original_gt[3], 
                                original_gt[4], 
                                target_pixel_size_y
                            )
                            projection = ds.GetProjection()
                            print(f"Using adjusted reference from {r}m to {resolution}m resolution: {reference_shape}")
                            print(f"Debug: Factor = {r}/{resolution} = {factor}")
                            print(f"Debug: Original size: {ds.RasterYSize}x{ds.RasterXSize}, Target size: {reference_shape}")
                            break
                        except Exception as e:
                            print(f"Error reading fallback file {ref_path}: {e}")
                if reference_shape:
                    break

        if reference_shape is None:
            print("Unable to determine reference size from any available data.")
            return False

        # Determine maximum position for output array
        max_position = max(pos for _, pos , _,_ in band_list)
        output_array = [None] * max_position
        band_names = [None] * max_position

        # Process each band
        for band_id, position, name,_ in band_list:
            # Memorizza il nome della banda nella posizione corretta
            band_names[position - 1] = name
            
            # if band_id is 9 we need to check for 8A
            if band_id == 9:
                band_str = "8A"
            # as a result we need to check for band_id minus 1 from band id greater than 9
            elif band_id > 9:
                # Check if band_id is 13 (SCL) and adjust accordingly
                if band_id == 14:
                    band_str = "SCL"
                else:
                    band_str = str(band_id-1).zfill(2)
            else:
                band_str = str(band_id).zfill(2)
        
            band_file = None
            
            # First try to find the band at the requested resolution
            name_patterns = [f"B{band_str}_{resolution}m", f"{band_str}_{resolution}m.jp2"]
            for pattern in name_patterns:
                matching_files = [f for f in files if pattern in f]
                
                if matching_files:
                    band_file = os.path.join(folder_path, matching_files[0])
                    print(f"Found band {band_id} at {resolution}m: {matching_files[0]}")
                    break

            if band_file and os.path.exists(band_file):
                try:
                    array = resample_with_gdal(band_file, reference_shape[1], reference_shape[0])
                    print(f"Successfully loaded band {band_id} from {band_file}")
                except Exception as e:
                    print(f"Error processing band {band_id}: {e}")
                    array = np.zeros(reference_shape, dtype=np.uint16)
            else:
                # Try other resolutions
                fallback_array = None
                for r in [10, 20, 60]:  # Try all resolutions, including the target one
                    if r == resolution:
                        continue
                    temp_folder = found_folder(input_path, r)
                    if not temp_folder:
                        continue
                        
                    temp_files = [f for f in os.listdir(temp_folder) if f.endswith('.jp2') or f.endswith('.tif')]
                    fallback_patterns = [f"B{band_str}_{r}m", f"{band_str}_{r}m.jp2"]
                    
                    for pattern in fallback_patterns:
                        matching_files = [f for f in temp_files if pattern in f]
                        if matching_files:
                            fallback_path = os.path.join(temp_folder, matching_files[0])
                            print(f"Using fallback for band {band_id}: {fallback_path}")
                            try:
                                if band_id == 8 and resolution < r:
                                    # Special handling for band 8 when upsampling
                                    arr, _ = read_gdal_array(fallback_path)
                                    factor = r / resolution  # CORREZIONE: r=dati disponibili, resolution=target
                                    # For upsampling, use proper resampling instead of naive repeat
                                    fallback_array = resample_with_gdal(
                                        fallback_path, 
                                        int(arr.shape[1] * factor), 
                                        int(arr.shape[0] * factor)
                                    )
                                    # Crop to reference size if needed
                                    fallback_array = fallback_array[:reference_shape[0], :reference_shape[1]]
                                else:
                                    # Normal resampling for other bands
                                    fallback_array = resample_with_gdal(fallback_path, reference_shape[1], reference_shape[0])
                                break
                            except Exception as e:
                                print(f"Error processing fallback for band {band_id}: {e}")
                                
                    if fallback_array is not None:
                        break
                        
                array = fallback_array if fallback_array is not None else np.zeros(reference_shape, dtype=np.uint16)
                
                if fallback_array is None:
                    print(f"Warning: Using zeros for band {band_id} - could not find data")

            # Ensure array is the right shape and dtype
            if array.shape != reference_shape:
                print(f"Warning: Reshaping band {band_id} from {array.shape} to {reference_shape}")
                # Resize array to match reference shape
                try:
                    from scipy.ndimage import zoom
                    zoom_factors = (reference_shape[0] / array.shape[0], reference_shape[1] / array.shape[1])
                    array = zoom(array, zoom_factors, order=1)
                    # Ensure exact dimensions
                    array = array[:reference_shape[0], :reference_shape[1]]
                except ImportError:
                    # Fallback if scipy not available
                    array = np.zeros(reference_shape, dtype=np.uint16)
                    
            output_array[position - 1] = array.astype(np.uint16)  # Ensure uint16 datatype

        # Check if all bands were processed
        if any(band is None for band in output_array):
            missing = [i+1 for i, band in enumerate(output_array) if band is None]
            print(f"Warning: Missing data for positions {missing}")
            # Fill missing bands with zeros
            for i in range(len(output_array)):
                if output_array[i] is None:
                    output_array[i] = np.zeros(reference_shape, dtype=np.uint16)
                # Assicurati che anche i nomi mancanti siano riempiti
                if band_names[i] is None:
                    band_names[i] = f"Band_{i+1}"

        # Stack all bands together
        stack = np.stack(output_array, axis=0)
        
        # CORREZIONE 3: Verifica finale della risoluzione
        print(f"Debug: Final output resolution should be: {resolution}m")
        print(f"Debug: Final pixel size in geotransform: {geo_transform[1]}m x {abs(geo_transform[5])}m")
        print(f"Debug: Final stack shape: {stack.shape}")
        print(f"Debug: Final reference shape: {reference_shape}")
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Save as GeoTIFF
            driver = gdal.GetDriverByName('GTiff')
            out_ds = driver.Create(output_path, reference_shape[1], reference_shape[0], max_position, gdal.GDT_UInt16)
            
            if out_ds is None:
                print(f"Error: Could not create output file {output_path}")
                return False
                
            out_ds.SetGeoTransform(geo_transform)
            out_ds.SetProjection(projection)

            for i in range(max_position):
                band = out_ds.GetRasterBand(i + 1)
                band.WriteArray(stack[i])
                band.SetDescription(band_names[i])
                band.FlushCache()

            out_ds.FlushCache()
            out_ds = None
            print(f"Stack successfully saved to: {output_path}")
            return True
        except Exception as e:
            print(f"Error saving output: {e}")
            return False
    
                        
    
    if scenario == 3:
        # print(band_list)

        # the path is present in the band list because each tuple is (origine, destinazione, path)
        
        list_path = [path for _, _, _, path in band_list]
        list_names = [name for _, _, name, _ in band_list] 
            
        reference_shape, geo_transform, projection, stacked_array, datatype,flag = check_source(list_path)
        
        if flag is False:
            return 4
        
        # save the stacked array
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(output_path, reference_shape[1], reference_shape[0], len(list_path), datatype)
        if out_ds is None:
            print(f"Error: Could not create output file {output_path}")
            return False
        
        out_ds.SetGeoTransform(geo_transform)
        out_ds.SetProjection(projection)
        
        for i in range(len(list_path)):
            band = out_ds.GetRasterBand(i + 1)
            band.WriteArray(stacked_array[i])
            # modifica del 1/07/2025 aggiungere il nome della banda
            band.SetDescription(list_names[i])
            band.FlushCache()
            
        out_ds.FlushCache()
        out_ds = None
        # print(f"Stack successfully saved to: {output_path}")
        # Close the datasets
        ds = None
        out_ds = None
    
    return True








#OLD VERSION SCENARIO 2

# scenario 2    
    # if scenario == 2:        
    #     reference_shape = None
    #     geo_transform = None
    #     projection = None
        
    #     # Find folder with requested resolution
    #     folder_path = found_folder(input_path, resolution)
    #     if not folder_path:
    #         print(f"No folder found with resolution {resolution}m")
    #         return False

    #     # Get list of available files
    #     files = [f for f in os.listdir(folder_path) if f.endswith('.jp2') or f.endswith('.tif')]
        
        
        
        
    #     # Find reference dimensions from any valid band
    #     for band_id, _, _ in band_list:
    #         # if band_id is 9 we need to check for 8A
    #         if band_id == 9:
    #             band_str = "8A"
    #         # as a result we need to check for band_id minus 1 from band id greater than 9
    #         elif band_id > 9:
    #             # Check if band_id is 13 (SCL) and adjust accordingly
    #             if band_id == 14:
    #                 band_str = "SCL"
    #             else:
    #                 band_str = str(band_id - 1).zfill(2)

    #         else:
    #             band_str = str(band_id).zfill(2)
            
    #         # Check both possible file naming patterns
    #         for name_pattern in [f"B{band_str}_{resolution}m", f"B{band_str}_{resolution}m.jp2", f"{band_str}_{resolution}m.jp2"]:
    #             matching_files = [f for f in files if name_pattern in f]
    #             if matching_files:
    #                 ref_path = os.path.join(folder_path, matching_files[0])
    #                 try:
    #                     array, ds = read_gdal_array(ref_path)
    #                     reference_shape = (ds.RasterYSize, ds.RasterXSize)
    #                     geo_transform = ds.GetGeoTransform()
    #                     projection = ds.GetProjection()
    #                     print(f"Found reference dimensions: {reference_shape} from {matching_files[0]}")
    #                     break
    #                 except Exception as e:
    #                     print(f"Error reading reference file {ref_path}: {e}")
    #         if reference_shape:
    #             break

    #     # If no reference found at requested resolution, try other resolutions
    #     if reference_shape is None:
    #         for r in [10, 20, 60]:
    #             if r == resolution:
    #                 continue
    #             temp_folder = found_folder(input_path, r)
    #             if temp_folder:
    #                 temp_files = [f for f in os.listdir(temp_folder) if f.endswith('.jp2') or f.endswith('.tif')]
    #                 if temp_files:
    #                     ref_path = os.path.join(temp_folder, temp_files[0])
    #                     try:
    #                         array, ds = read_gdal_array(ref_path)
    #                         # Adjust dimensions for resolution difference
    #                         factor = resolution / r
    #                         reference_shape = (int(ds.RasterYSize * factor), int(ds.RasterXSize * factor))
    #                         # Adjust geotransform for new resolution
    #                         original_gt = ds.GetGeoTransform()
    #                         geo_transform = (
    #                             original_gt[0], original_gt[1] / factor, original_gt[2],
    #                             original_gt[3], original_gt[4], original_gt[5] / factor
    #                         )
    #                         projection = ds.GetProjection()
    #                         print(f"Using adjusted reference from {r}m resolution: {reference_shape}")
    #                         break
    #                     except Exception as e:
    #                         print(f"Error reading fallback file {ref_path}: {e}")
    #             if reference_shape:
    #                 break

    #     if reference_shape is None:
    #         print("Unable to determine reference size from any available data.")
    #         return False

    #     # Determine maximum position for output array
    #     max_position = max(pos for _, pos , _ in band_list)
    #     output_array = [None] * max_position

    #     # Process each band
    #     for band_id, position, _ in band_list:
    #         # if band_id is 9 we need to check for 8A
    #         if band_id == 9:
    #             band_str = "8A"
    #         # as a result we need to check for band_id minus 1 from band id greater than 9
    #         elif band_id > 9:
    #             # Check if band_id is 13 (SCL) and adjust accordingly
    #             if band_id == 14:
    #                 band_str = "SCL"
    #             else:
    #                 band_str = str(band_id - 1).zfill(2)
    #         else:
    #             band_str = str(band_id).zfill(2)
           
    #         band_file = None
            
    #         # First try to find the band at the requested resolution
    #         name_patterns = [f"B{band_str}_{resolution}m", f"{band_str}_{resolution}m.jp2"]
    #         for pattern in name_patterns:
    #             matching_files = [f for f in files if pattern in f]
    #             if matching_files:
    #                 band_file = os.path.join(folder_path, matching_files[0])
    #                 print(f"Found band {band_id} at {resolution}m: {matching_files[0]}")
    #                 break

    #         if band_file and os.path.exists(band_file):
    #             try:
    #                 array = resample_with_gdal(band_file, reference_shape[1], reference_shape[0])
    #                 print(f"Successfully loaded band {band_id} from {band_file}")
    #             except Exception as e:
    #                 print(f"Error processing band {band_id}: {e}")
    #                 array = np.zeros(reference_shape, dtype=np.uint16)
    #         else:
    #             # Try other resolutions
    #             fallback_array = None
    #             for r in [10, 20, 60]:  # Try all resolutions, including the target one
    #                 if r == resolution:
    #                     continue
    #                 temp_folder = found_folder(input_path, r)
    #                 if not temp_folder:
    #                     continue
                        
    #                 temp_files = [f for f in os.listdir(temp_folder) if f.endswith('.jp2') or f.endswith('.tif')]
    #                 fallback_patterns = [f"B{band_str}_{r}m", f"{band_str}_{r}m.jp2"]
                    
    #                 for pattern in fallback_patterns:
    #                     matching_files = [f for f in temp_files if pattern in f]
    #                     if matching_files:
    #                         fallback_path = os.path.join(temp_folder, matching_files[0])
    #                         print(f"Using fallback for band {band_id}: {fallback_path}")
    #                         try:
    #                             if band_id == 8 and resolution < r:
    #                                 # Special handling for band 8 when upsampling
    #                                 arr, _ = read_gdal_array(fallback_path)
    #                                 factor = r / resolution
    #                                 # For upsampling, use proper resampling instead of naive repeat
    #                                 fallback_array = resample_with_gdal(
    #                                     fallback_path, 
    #                                     int(arr.shape[1] * factor), 
    #                                     int(arr.shape[0] * factor)
    #                                 )
    #                                 # Crop to reference size if needed
    #                                 fallback_array = fallback_array[:reference_shape[0], :reference_shape[1]]
    #                             else:
    #                                 # Normal resampling for other bands
    #                                 fallback_array = resample_with_gdal(fallback_path, reference_shape[1], reference_shape[0])
    #                             break
    #                         except Exception as e:
    #                             print(f"Error processing fallback for band {band_id}: {e}")
                                
    #                 if fallback_array is not None:
    #                     break
                        
    #             array = fallback_array if fallback_array is not None else np.zeros(reference_shape, dtype=np.uint16)
                
    #             if fallback_array is None:
    #                 print(f"Warning: Using zeros for band {band_id} - could not find data")

    #         # Ensure array is the right shape and dtype
    #         if array.shape != reference_shape:
    #             print(f"Warning: Reshaping band {band_id} from {array.shape} to {reference_shape}")
    #             # Resize array to match reference shape
    #             try:
    #                 from scipy.ndimage import zoom
    #                 zoom_factors = (reference_shape[0] / array.shape[0], reference_shape[1] / array.shape[1])
    #                 array = zoom(array, zoom_factors, order=1)
    #                 # Ensure exact dimensions
    #                 array = array[:reference_shape[0], :reference_shape[1]]
    #             except ImportError:
    #                 # Fallback if scipy not available
    #                 array = np.zeros(reference_shape, dtype=np.uint16)
                    
    #         output_array[position - 1] = array.astype(np.uint16)  # Ensure uint16 datatype

    #     # Check if all bands were processed
    #     if any(band is None for band in output_array):
    #         missing = [i+1 for i, band in enumerate(output_array) if band is None]
    #         print(f"Warning: Missing data for positions {missing}")
    #         # Fill missing bands with zeros
    #         for i in range(len(output_array)):
    #             if output_array[i] is None:
    #                 output_array[i] = np.zeros(reference_shape, dtype=np.uint16)

    #     # Stack all bands together
    #     stack = np.stack(output_array, axis=0)
        
    #     try:
    #         # Create output directory if it doesn't exist
    #         os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
    #         # Save as GeoTIFF
    #         driver = gdal.GetDriverByName('GTiff')
    #         out_ds = driver.Create(output_path, reference_shape[1], reference_shape[0], max_position, gdal.GDT_UInt16)
            
    #         if out_ds is None:
    #             print(f"Error: Could not create output file {output_path}")
    #             return False
                
    #         out_ds.SetGeoTransform(geo_transform)
    #         out_ds.SetProjection(projection)

    #         for i in range(max_position):
    #             band = out_ds.GetRasterBand(i + 1)
    #             band.WriteArray(stack[i])
    #             band.FlushCache()

    #         out_ds.FlushCache()
    #         out_ds = None
    #         print(f"Stack successfully saved to: {output_path}")
    #         return True
    #     except Exception as e:
    #         print(f"Error saving output: {e}")
    #         return False