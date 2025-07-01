import re
import numpy as np
from osgeo import gdal
from qgis.PyQt.QtWidgets import QAction, QFileDialog, QMessageBox
import os
from qgis.core import QgsExpression


"""
Indicies Costum TAB 
Function to check if the expression is valid
"""
def has_invalid_operators(expr):
    # Trova sequenze di due o più operatori, tranne `**`
    pattern = r'(?<!\*)[\+\-\*/%]{2,}(?!\*)'  # esclude ** ma trova ++, --, //, etc.
    return re.search(pattern, expr) is not None

def has_zero_division(expr):
    # Trova divisioni per zero
    pattern = r'/\s*0\b'
    return re.search(pattern, expr) is not None

def has_adjacent_band_names(expr):
    return re.search(r'(B\d{1,2})(B\d{1,2})', expr, flags=re.IGNORECASE) is not None

def has_invalid_characters(expr):
    # Define a regex pattern to match invalid characters
    # ACCETTA SOLO LA LETTERA B seguita da un numero
    # e i seguenti operatori: + - * / ( ) ^
    pattern = r'[^B\d\s\+\-\*/\(\)\^]'
    return re.search(pattern, expr) is not None

def has_invalid_band_names(expr):
    # Cerca qualsiasi parola che inizia con B, ma NON è B seguita da 1 o 2 cifre
    pattern = r'\bB(?!\d{1,2}\b)[A-Za-z0-9]*'
    return re.search(pattern, expr, flags=re.IGNORECASE) is not None

def save_band_to_file(output_path, band_data, dataset):
    """
    Save the band data to a GeoTIFF file.
    """
    # Create a new GeoTIFF file
    driver = gdal.GetDriverByName('GTiff')
    output_dataset = driver.Create(output_path, dataset.RasterXSize, dataset.RasterYSize, 1, gdal.GDT_Float32)
    
    # Set the geotransform and projection from the input dataset
    output_dataset.SetGeoTransform(dataset.GetGeoTransform())
    output_dataset.SetProjection(dataset.GetProjection())
    
    # Write the band data to the output file
    output_dataset.GetRasterBand(1).WriteArray(band_data)
    
    # Close the dataset
    output_dataset.FlushCache()
    output_dataset = None
    return

def compute_vegetation_indices(input_path, band_list, indices_list, scale_factor,L_SAVI):
    
    """
    """
    # read the input file
    
    dataset = gdal.Open(input_path)
    # print(indices_list)
    
    for element in indices_list:
        index_name = element[0]
        index_out_path = element[1]
    
        if index_name == 'NDVI':
            # (NIR-RED)/(NIR+RED)
            # NIR = B8
            # RED = B4
            
            # from band_list search for band 4 and band 8 and get the position
            
            B8_position = [i[1] for i in band_list if i[0] == 8][0]
            B4_position = [i[1] for i in band_list if i[0] == 4][0]
            

            # Get the bands for NDVI from dataset
            B8 = dataset.GetRasterBand(B8_position).ReadAsArray().astype(np.float32)
            B4 = dataset.GetRasterBand(B4_position).ReadAsArray().astype(np.float32)
            
            # Apply scale factor
            B8 = B8 / scale_factor
            B4 = B4 / scale_factor

            # Calculate denominator
            denominator = B8 + B4

            # Calculate NDVI with proper zero division handling
            with np.errstate(divide='ignore', invalid='ignore'):
                NDVI = (B8 - B4) / denominator
                NDVI[denominator == 0] = np.nan
            
            # Save the NDVI band to the output file           
            save_band_to_file(index_out_path, NDVI, dataset)
 
        if index_name == 'EVI':
            # 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)
            # NIR = B8
            # RED = B4
            # BLUE = B2
            
            # from band_list search for band 4 and band 8 and get the position
            
            B8_position = [i[1] for i in band_list if i[0] == 8][0]
            B4_position = [i[1] for i in band_list if i[0] == 4][0]
            B2_position = [i[1] for i in band_list if i[0] == 2][0]
            
            # Get the bands for NDVI from dataset
            B8 = dataset.GetRasterBand(B8_position).ReadAsArray().astype(np.float64)
            B4 = dataset.GetRasterBand(B4_position).ReadAsArray().astype(np.float64)
            B2 = dataset.GetRasterBand(B2_position).ReadAsArray().astype(np.float64)
            
            # Apply scale factor
            B8 = B8 / scale_factor
            B4 = B4 / scale_factor
            B2 = B2 / scale_factor
            
            # Calculate denominator
            denominator = B8 + 6 * B4 - 7.5 * B2 + 1
            numerator = 2.5 * (B8 - B4)

            # Calculate EVI with proper zero division handling
            with np.errstate(divide='ignore', invalid='ignore'):
                EVI = numerator / denominator
                EVI[denominator == 0] = np.nan 
                
            # Save the EVI band to the output file
            save_band_to_file(index_out_path, EVI, dataset)
        
        if index_name == 'SAVI':
            # (NIR-RED)/(NIR+RED+L)*(1+L)
            # NIR = B8
            # RED = B4
            # L = 0.5
            
            # from band_list search for band 4 and band 8 and get the position
            
            B8_position = [i[1] for i in band_list if i[0] == 8][0]
            B4_position = [i[1] for i in band_list if i[0] == 4][0]
            
            # Get the bands for NDVI from dataset
            B8 = dataset.GetRasterBand(B8_position).ReadAsArray().astype(np.float32)
            B4 = dataset.GetRasterBand(B4_position).ReadAsArray().astype(np.float32)
            
            # Apply scale factor
            B8 = B8 / scale_factor
            B4 = B4 / scale_factor
            
            # Calculate SAVI with proper zero division handling
            L = L_SAVI
            denominator = B8 + B4 + L

            with np.errstate(divide='ignore', invalid='ignore'):
                SAVI = ((B8 - B4) / denominator) * (1 + L)
                SAVI[denominator == 0] = np.nan 
            
            # Save the SAVI band to the output file
            save_band_to_file(index_out_path, SAVI, dataset)
        
    
        if index_name == 'NDRE':
            # (NIR - RED_ED) / (NIR + RED_ED)
            # NIR = B8
            # REDEDGE = B5
            
            # from band_list search for band 5 and band 8 and get the position
            
            B8_position = [i[1] for i in band_list if i[0] == 8][0]
            B5_position = [i[1] for i in band_list if i[0] == 5][0]
            
            # Get the bands for NDVI from dataset
            B8 = dataset.GetRasterBand(B8_position).ReadAsArray().astype(np.float32)
            B5 = dataset.GetRasterBand(B5_position).ReadAsArray().astype(np.float32)
            
            # Apply scale factor
            B8 = B8 / scale_factor
            B5 = B5 / scale_factor
            
            # Calculate denominator
            denominator = B8 + B5

            # Calculate NDRE with proper zero division handling
            with np.errstate(divide='ignore', invalid='ignore'):
                NDRE = (B8 - B5) / denominator
                NDRE[denominator == 0] = np.nan 
            
            # Save the NDRE band to the output file
            save_band_to_file(index_out_path, NDRE, dataset)
    
    return 

def compute_water_indices(input_path, band_list, indices_list, scale_factor):
    
    """
    """
    # read the input file
    
    dataset = gdal.Open(input_path)
    # print(indices_list)
    
    for element in indices_list:
        index_name = element[0]
        index_out_path = element[1]
    
        if index_name == 'NDWI':
            # (GREEN-NIR)/(GREEN+NIR)
            # NIR = B8
            # GREEN = B3
            
            # from band_list search for band 3 and band 8 and get the position
            
            B8_position = [i[1] for i in band_list if i[0] == 8][0]
            B3_position = [i[1] for i in band_list if i[0] == 3][0]
            

            # Get the bands for NDVI from dataset
            B8 = dataset.GetRasterBand(B8_position).ReadAsArray().astype(np.float32)
            B3 = dataset.GetRasterBand(B3_position).ReadAsArray().astype(np.float32)
            
            # Apply scale factor
            B8 = B8 / scale_factor
            B3 = B3 / scale_factor

            # Calculate denominator
            denominator = B8 + B3

            # Calculate NDVI with proper zero division handling
            with np.errstate(divide='ignore', invalid='ignore'):
                NDWI = (B3 - B8) / denominator
                NDWI[denominator == 0] = np.nan
            
            # Save the NDWI band to the output file           
            save_band_to_file(index_out_path, NDWI, dataset)
 
        if index_name == 'MNDWI':
            # (GREEN - SWIR2) / (GREEN+SWIR2)
            # GREEN = B3
            # SWIR2 = B12
            
            # from band_list search for band 3 and band 12 and get the position
            
            B3_position = [i[1] for i in band_list if i[0] == 3][0]
            B12_position = [i[1] for i in band_list if i[0] == 13][0]
           
            
            # Get the bands for MNDWI from dataset
            B3 = dataset.GetRasterBand(B3_position).ReadAsArray().astype(np.float32)
            B12 = dataset.GetRasterBand(B12_position).ReadAsArray().astype(np.float32)
            
            # Apply scale factor
            B3 = B3 / scale_factor
            B12 = B12 / scale_factor
            
            # Calculate denominator
            denominator = B3 + B12
            numerator = B3 - B12
            
            # Calculate MNDWI with proper zero division handling
            with np.errstate(divide='ignore', invalid='ignore'):
                MNDWI = numerator / denominator
                MNDWI[denominator == 0] = np.nan 
                
            # Save the MNDWI band to the output file
            save_band_to_file(index_out_path, MNDWI, dataset)
        
        if index_name == 'NDMI':
            # (NIR - SWIR1) / (NIR + SWIR1)
            # NIR = B8
            # SWIR1 = B11

            # from band_list search for band 11 and band 8 and get the position
            
            B8_position = [i[1] for i in band_list if i[0] == 8][0]
            B11_position = [i[1] for i in band_list if i[0] == 12][0]
            
            # Get the bands for NDMI from dataset
            B8 = dataset.GetRasterBand(B8_position).ReadAsArray().astype(np.float32)
            B11 = dataset.GetRasterBand(B11_position).ReadAsArray().astype(np.float32)
            
            # Apply scale factor
            B8 = B8 / scale_factor
            B11 = B11 / scale_factor
            
            # Calculate NDMI with proper zero division handling
            denominator = B8 + B11

            with np.errstate(divide='ignore', invalid='ignore'):
                NDMI = ((B8 - B11) / denominator) 
                NDMI[denominator == 0] = np.nan 
            
            # Save the NDMI band to the output file
            save_band_to_file(index_out_path, NDMI, dataset)
        
    return 

def compute_fire_indices(input_path, band_list, indices_list, scale_factor):
    
    """
    """
    # read the input file
    
    dataset = gdal.Open(input_path)
    # print(indices_list)
    
    for element in indices_list:
        index_name = element[0]
        index_out_path = element[1]
    
        if index_name == 'NBR':
            # (NIR-SWIR2)/(NIR+SWIR2)
            # NIR = B8
            # SWIR2= B12
            
            # from band_list search for band 12 and band 8 and get the position
            
            B8_position = [i[1] for i in band_list if i[0] == 8][0]
            B12_position = [i[1] for i in band_list if i[0] == 13][0]
            

            # Get the bands for NBR from dataset
            B8 = dataset.GetRasterBand(B8_position).ReadAsArray().astype(np.float32)
            B12 = dataset.GetRasterBand(B12_position).ReadAsArray().astype(np.float32)
            
            # Apply scale factor
            B8 = B8 / scale_factor
            B12 = B12 / scale_factor

            # Calculate denominator
            denominator = B8 + B12

            # Calculate NBR with proper zero division handling
            with np.errstate(divide='ignore', invalid='ignore'):
                NBR = (B8 - B12) / denominator
                NBR[denominator == 0] = np.nan
            
            # Save the NDWI band to the output file           
            save_band_to_file(index_out_path, NBR, dataset)
 
        if index_name == 'NBR2':
            # (SWIR2 - SWIR1) / (SWIR2+SWIR1)
            # SWIR1 = B11
            # SWIR2 = B12
            
            # from band_list search for band 11 and band 12 and get the position
            
            B11_position = [i[1] for i in band_list if i[0] == 12][0]
            B12_position = [i[1] for i in band_list if i[0] == 13][0]
           
            
            # Get the bands for NBR2 from dataset
            B11 = dataset.GetRasterBand(B11_position).ReadAsArray().astype(np.float32)
            B12 = dataset.GetRasterBand(B12_position).ReadAsArray().astype(np.float32)
            
            # Apply scale factor
            B11 = B11 / scale_factor
            B12 = B12 / scale_factor
            
            # Calculate denominator
            denominator = B11 + B12
            numerator = B12 - B11
            
            # Calculate NBR2 with proper zero division handling
            with np.errstate(divide='ignore', invalid='ignore'):
                NBR2 = numerator / denominator
                NBR2[denominator == 0] = np.nan 
                
            # Save the NBR2 band to the output file
            save_band_to_file(index_out_path, NBR2, dataset)
        
        if index_name == 'MIRBI':
            # 10*SWIR2 -9.8*SWIR11 + 2
            # SWIR1 = B11
            # SWIR2 = B12
            
            # from band_list search for band 11 and band 12 and get the position
            
            B12_position = [i[1] for i in band_list if i[0] == 13][0]
            B11_position = [i[1] for i in band_list if i[0] == 12][0]
            
            # Get the bands for MIRBI from dataset
            B12 = dataset.GetRasterBand(B12_position).ReadAsArray().astype(np.float32)
            B11 = dataset.GetRasterBand(B11_position).ReadAsArray().astype(np.float32)
            
            # Apply scale factor
            B12 = B12 / scale_factor
            B11 = B11 / scale_factor
            
            # Calculate MIRBI 
            MIRBI = 10*B12-9.8*B11+2
            
            # Save the MIRBI band to the output file
            save_band_to_file(index_out_path, MIRBI, dataset)
        
    return 


def compute_building_indices(input_path, band_list, indices_list, scale_factor):
    
    """
    """
    # read the input file
    
    dataset = gdal.Open(input_path)
    # print(indices_list)
    
    for element in indices_list:
        index_name = element[0]
        index_out_path = element[1]
    
        if index_name == 'NDBI':
            # (SWIR2-NIR)/(SWIR2+NIR)
            # NIR = B8
            # SWIR2= B12
            
            # from band_list search for band 12 and band 8 and get the position
            
            B8_position = [i[1] for i in band_list if i[0] == 8][0]
            B12_position = [i[1] for i in band_list if i[0] == 13][0]
            

            # Get the bands for NDBI from dataset
            B8 = dataset.GetRasterBand(B8_position).ReadAsArray().astype(np.float32)
            B12 = dataset.GetRasterBand(B12_position).ReadAsArray().astype(np.float32)
            
            # Apply scale factor
            B8 = B8 / scale_factor
            B12 = B12 / scale_factor

            # Calculate denominator
            denominator = B8 + B12

            # Calculate NDBI with proper zero division handling
            with np.errstate(divide='ignore', invalid='ignore'):
                NDBI = (B12 - B8) / denominator
                NDBI[denominator == 0] = np.nan
            
            # Save the NDBI band to the output file           
            save_band_to_file(index_out_path, NDBI, dataset)
 
        if index_name == 'NBI':
            # (SWIR2 * RED) / (NIR)
            # SWIR2 = B12
            # RED = B4
            # NIR = B8
            
            # from band_list search for band 11 and band 12 and get the position
            
            B4_position = [i[1] for i in band_list if i[0] == 4][0]
            B8_position = [i[1] for i in band_list if i[0] == 8][0]
            B12_position = [i[1] for i in band_list if i[0] == 13][0]
           
            
            # Get the bands for NBI from dataset
            B4 = dataset.GetRasterBand(B4_position).ReadAsArray().astype(np.float32)
            B8 = dataset.GetRasterBand(B8_position).ReadAsArray().astype(np.float32)
            B12 = dataset.GetRasterBand(B12_position).ReadAsArray().astype(np.float32)
            
            # Apply scale factor
            B4 = B4 / scale_factor
            B8 = B8 / scale_factor
            B12 = B12 / scale_factor
            
            # Calculate denominator
            denominator =  B8 
            numerator = B4*B12
            
            # Calculate NBI with proper zero division handling
            with np.errstate(divide='ignore', invalid='ignore'):
                NBI = numerator / denominator
                NBI[denominator == 0] = np.nan 
                
            # Save the NBI band to the output file
            save_band_to_file(index_out_path, NBI, dataset)
        
        if index_name == 'NBAI':
            # [(SWIR2-SWIR1)/GREEN]/[(SWIR2+SWIR1)/GREEN]
            # GREEN = B3
            # SWIR1 = B11
            # SWIR2 = B12
            
            # from band_list search for band 11 and band 12 and band 3 and get the position
            
            B3_position = [i[1] for i in band_list if i[0] == 3][0]
            B12_position = [i[1] for i in band_list if i[0] == 13][0]
            B11_position = [i[1] for i in band_list if i[0] == 12][0]
            
            # Get the bands for NBAI from dataset
            B3 = dataset.GetRasterBand(B3_position).ReadAsArray().astype(np.float32)
            B12 = dataset.GetRasterBand(B12_position).ReadAsArray().astype(np.float32)
            B11 = dataset.GetRasterBand(B11_position).ReadAsArray().astype(np.float32)
            
            # Apply scale factor
            B3 = B3 / scale_factor
            B12 = B12 / scale_factor
            B11 = B11 / scale_factor
            
            # Calculate denominator
            numerator = (B12-B11)/B3
            denominator = (B12+B11)/B3
            
            
            # Calculate NBAI with proper zero division handling
            with np.errstate(divide='ignore', invalid='ignore'):
                NBAI = numerator / denominator
                NBAI[denominator == 0] = np.nan 
            
            # Save the MIRNBAIBI band to the output file
            save_band_to_file(index_out_path, NBAI, dataset)
            
        if index_name == 'BAEI':
            # (RED+0.3)/(GREEN + SWIR1)
            # RED = B4
            # GREEN = B3
            # SWIR1 = B11
                        
            # from band_list search for band 11 and band 4 and band 3 and get the position
            
            B3_position = [i[1] for i in band_list if i[0] == 3][0]
            B4_position = [i[1] for i in band_list if i[0] == 4][0]
            B11_position = [i[1] for i in band_list if i[0] == 12][0]
            
            # Get the bands for BAEI from dataset
            B3 = dataset.GetRasterBand(B3_position).ReadAsArray().astype(np.float32)
            B4 = dataset.GetRasterBand(B4_position).ReadAsArray().astype(np.float32)
            B11 = dataset.GetRasterBand(B11_position).ReadAsArray().astype(np.float32)
            
            # Apply scale factor
            B3 = B3 / scale_factor
            B4 = B4 / scale_factor
            B11 = B11 / scale_factor
            
            # Calculate denominator
            numerator = B4+0.3
            denominator = B3+B11
            
            
            # Calculate BAEI with proper zero division handling
            with np.errstate(divide='ignore', invalid='ignore'):
                BAEI = numerator / denominator
                BAEI[denominator == 0] = np.nan 
            
            # Save the MIRNBAIBI band to the output file
            save_band_to_file(index_out_path, BAEI, dataset)
        
    return 



def compute_expression( input_path, band_list, expression, scale_factor, output_path):
    
    # open the input file
    dataset = gdal.Open(input_path)

    # upload the bands in a dictionary
    bands_data = {}
    
    for element in band_list:
        band_name = f'B{element[0]}'
        band_index = element[1]
        
        band = dataset.GetRasterBand(band_index)
        
        
        if band is None:
            raise RuntimeError(f"Banda {band_name} (band_index {band_index}) not found.")
        
        bands_data[band_name] = band.ReadAsArray().astype(np.float32)/scale_factor


    # evaluate the expression
    result = eval(expression, {"__builtins__": {}}, bands_data)
    
    # Save the result to a GeoTIFF file
    save_band_to_file(output_path, result, dataset)   
    
    
    return