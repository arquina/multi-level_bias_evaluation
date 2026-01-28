import torch
import os
import pandas as pd
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
from shapely import Point, Polygon
import h5py
import numpy as np
import argparse

def is_point_in_polygon(point, polygons):
    """
    Checks if a point is in any independent polygon and outside any inner polygon
    associated with it.

    Parameters:
    - point: Tuple representing the (x, y) coordinates of the point to check.
    - polygons: List of tuples, where each tuple has an independent polygon as the 
                first element and a list of inner polygons (holes) as the second element.

    Returns:
    - True if the point is in an independent polygon and not in its inner polygons,
      False otherwise.
    """
    point_geom = Point(point)
    
    for independent_polygon, inner_polygons in polygons:
        # Check if point is in the independent polygon
        if independent_polygon.contains(point_geom):
            # Check if point is in any of the inner polygons
            if any(inner.contains(point_geom) for inner in inner_polygons):
                return False  # Point is in an inner polygon, so exclude it
            return True  # Point is in the independent polygon and not in any inner polygons
    
    return False

def organize_polygons(polygons):
    """
    Organizes polygons into a structure where independent polygons are paired with any inner polygons (holes).
    
    Parameters:
    - polygons: List of shapely Polygon objects.
    
    Returns:
    - List of tuples, where each tuple consists of an independent polygon and a list of inner polygons (holes).
    """
    independent_polygons = []
    
    for outer in polygons:
        inner_polygons = []
        for inner in polygons:
            if outer != inner and outer.contains(inner):
                inner_polygons.append(inner)
        
        # Only add to independent_polygons if it has no containing polygons
        if not any(other.contains(outer) for other in polygons if other != outer):
            independent_polygons.append((outer, inner_polygons))
    
    return independent_polygons

def extract_feature(save_dir, feature_dir, metadata, pfm_list, xml_root_dir = None, extract_stainnorm = False, cancer_only = False):
    metadata = pd.read_csv(metadata)
    patch_dict = {'virchow': '224', 'virchow2': '224', 'UNI': '256', 'UNI2': '256', 'gigapath': '256', 'CONCH': '512'}
    os.makedirs(save_dir, exist_ok = True)

    whole_cancer_save_dir = os.path.join(save_dir, 'whole_cancer')
    os.makedirs(whole_cancer_save_dir, exist_ok = True)

    if cancer_only:
        cancer_only_save_dir = os.path.join(save_dir, 'cancer_only')
        non_cancer_save_dir = os.path.join(save_dir, 'non_cancer')
        os.makedirs(cancer_only_save_dir, exist_ok = True)
        os.makedirs(non_cancer_save_dir, exist_ok = True)
    
    for pfm in pfm_list:
        patch_size = patch_dict[pfm]
        if pfm == 'UNI':
            pfm = 'uni_v1'
        elif pfm == 'UNI2':
            pfm = 'uni_v2'
        elif pfm == 'CONCH':
            pfm = 'conch_v1'

        whole_cancer_target_feature_dir = os.path.join(whole_cancer_save_dir, pfm)
        os.makedirs(whole_cancer_target_feature_dir, exist_ok = True)

        if cancer_only:
            cancer_only_target_feature_dir = os.path.join(cancer_only_save_dir, pfm)
            non_cancer_target_feature_dir = os.path.join(non_cancer_save_dir, pfm)
            os.makedirs(cancer_only_target_feature_dir, exist_ok = True)
            os.makedirs(non_cancer_target_feature_dir, exist_ok = True)

        whole_cancer_patch_dir = os.path.join(whole_cancer_target_feature_dir, patch_size)
        os.makedirs(whole_cancer_patch_dir, exist_ok = True)
        
        if cancer_only:
            cancer_only_patch_dir = os.path.join(cancer_only_target_feature_dir, patch_size)
            non_cancer_patch_dir = os.path.join(non_cancer_target_feature_dir, patch_size)
            os.makedirs(cancer_only_patch_dir, exist_ok = True)
            os.makedirs(non_cancer_patch_dir, exist_ok = True)

        if extract_stainnorm:
            sn_whole_cancer_target_feature_dir = os.path.join(whole_cancer_save_dir, pfm + '_stainnorm')
            os.makedirs(sn_whole_cancer_target_feature_dir, exist_ok = True)
            if cancer_only:
                sn_cancer_only_target_feature_dir = os.path.join(cancer_only_save_dir, pfm + '_stainnorm')
                sn_non_cancer_target_feature_dir = os.path.join(non_cancer_save_dir, pfm + '_stainnorm')
                os.makedirs(sn_cancer_only_target_feature_dir, exist_ok = True)
                os.makedirs(sn_non_cancer_target_feature_dir, exist_ok = True)

            sn_whole_cancer_patch_dir = os.path.join(sn_whole_cancer_target_feature_dir, patch_size)
            os.makedirs(sn_whole_cancer_patch_dir, exist_ok = True)

            if cancer_only:
                sn_cancer_only_patch_dir = os.path.join(sn_cancer_only_target_feature_dir, patch_size)
                sn_non_cancer_patch_dir = os.path.join(sn_non_cancer_target_feature_dir, patch_size)
                os.makedirs(sn_cancer_only_patch_dir, exist_ok = True)
                os.makedirs(sn_non_cancer_patch_dir, exist_ok = True)

        whole_cancer_total_feature_save_dir = os.path.join(whole_cancer_patch_dir, 'total_feature')
        whole_cancer_mean_feature_save_dir = os.path.join(whole_cancer_patch_dir, 'mean_feature')
        whole_cancer_coords_save_dir = os.path.join(whole_cancer_patch_dir, 'coords')

        os.makedirs(whole_cancer_total_feature_save_dir, exist_ok = True)
        os.makedirs(whole_cancer_mean_feature_save_dir, exist_ok = True)
        os.makedirs(whole_cancer_coords_save_dir, exist_ok = True)

        if cancer_only:
            cancer_only_total_feature_save_dir = os.path.join(cancer_only_patch_dir, 'total_feature')
            cancer_only_mean_feature_save_dir = os.path.join(cancer_only_patch_dir, 'mean_feature')
            cancer_only_coords_save_dir = os.path.join(cancer_only_patch_dir, 'coords')
            
            non_cancer_total_feature_save_dir = os.path.join(non_cancer_patch_dir, 'total_feature')
            non_cancer_mean_feature_save_dir = os.path.join(non_cancer_patch_dir, 'mean_feature')
            non_cancer_coords_save_dir = os.path.join(non_cancer_patch_dir, 'coords')

            os.makedirs(cancer_only_total_feature_save_dir, exist_ok = True)
            os.makedirs(cancer_only_mean_feature_save_dir, exist_ok = True)
            os.makedirs(cancer_only_coords_save_dir, exist_ok = True)

            os.makedirs(non_cancer_total_feature_save_dir, exist_ok = True)
            os.makedirs(non_cancer_mean_feature_save_dir, exist_ok = True)
            os.makedirs(non_cancer_coords_save_dir, exist_ok = True)
        
        if extract_stainnorm:
            sn_whole_cancer_total_feature_save_dir = os.path.join(sn_whole_cancer_patch_dir, 'total_feature')
            sn_whole_cancer_mean_feature_save_dir = os.path.join(sn_whole_cancer_patch_dir, 'mean_feature')
            sn_whole_cancer_coords_save_dir = os.path.join(sn_whole_cancer_patch_dir, 'coords')

            os.makedirs(sn_whole_cancer_total_feature_save_dir, exist_ok = True)
            os.makedirs(sn_whole_cancer_mean_feature_save_dir, exist_ok = True)
            os.makedirs(sn_whole_cancer_coords_save_dir, exist_ok = True)

            if cancer_only:
                sn_cancer_only_total_feature_save_dir = os.path.join(sn_cancer_only_patch_dir, 'total_feature')
                sn_cancer_only_mean_feature_save_dir = os.path.join(sn_cancer_only_patch_dir, 'mean_feature')
                sn_cancer_only_coords_save_dir = os.path.join(sn_cancer_only_patch_dir, 'coords')
                
                sn_non_cancer_total_feature_save_dir = os.path.join(sn_non_cancer_patch_dir, 'total_feature')
                sn_non_cancer_mean_feature_save_dir = os.path.join(sn_non_cancer_patch_dir, 'mean_feature')
                sn_non_cancer_coords_save_dir = os.path.join(sn_non_cancer_patch_dir, 'coords')
                
                os.makedirs(sn_cancer_only_total_feature_save_dir, exist_ok = True)
                os.makedirs(sn_cancer_only_mean_feature_save_dir, exist_ok = True)
                os.makedirs(sn_cancer_only_coords_save_dir, exist_ok = True)

                os.makedirs(sn_non_cancer_total_feature_save_dir, exist_ok = True)
                os.makedirs(sn_non_cancer_mean_feature_save_dir, exist_ok = True)
                os.makedirs(sn_non_cancer_coords_save_dir, exist_ok = True)
    
        with tqdm(total = len(metadata)) as pbar:
            for i, row in metadata.iterrows():
                database = row['database']
                tumor_subtype = row['subtype']
                subtype_name = row['subtype_name']
                file_name = row['feature_file_name']
                center = row['center']

                slide_id = file_name.split('.h5')[0]

                if database == 'TCGA':
                    subtype_feature_dir = os.path.join(feature_dir, database, subtype_name, '20x_' + str(patch_size) + 'px_0px_overlap', 'features_' + pfm)
                    sn_subtype_feature_dir = os.path.join(feature_dir, database, subtype_name, '20x_' + str(patch_size) + 'px_0px_overlap', 'features_' + pfm + '_stainnorm')
                else:
                    subtype_feature_dir = os.path.join(feature_dir, center, subtype_name, '20x_' + str(patch_size) + 'px_0px_overlap', 'features_' + pfm)
                    sn_subtype_feature_dir = os.path.join(feature_dir, center, subtype_name, '20x_' + str(patch_size) + 'px_0px_overlap', 'features_' + pfm + '_stainnorm')

                try:
                    whole_data = h5py.File(os.path.join(subtype_feature_dir, file_name))
                except:
                    print(os.path.join(subtype_feature_dir, file_name))
                    print(f'No data of {slide_id}')
                    continue

                print(tumor_subtype, slide_id)

                feature = torch.tensor(whole_data['features'][()])
                coords = whole_data['coords'][()]

                whole_mean_feature = torch.mean(feature, axis=0)
                torch.save(feature, os.path.join(whole_cancer_total_feature_save_dir, slide_id + '.pt'))
                torch.save(whole_mean_feature, os.path.join(whole_cancer_mean_feature_save_dir, slide_id + '.pt'))
                np.save(os.path.join(whole_cancer_coords_save_dir, slide_id + '.npy'), coords)

                if extract_stainnorm:
                    try:
                        sn_data = h5py.File(os.path.join(sn_subtype_feature_dir, file_name), 'r')
                    except:
                        print(f'No data of {slide_id}_stainnorm')
                        continue
                    sn_feature = torch.tensor(sn_data['features'][()])
                    sn_coords = sn_data['coords'][()]

                    coords_tuples = list(map(tuple, coords))
                    sn_coords_tuples = list(map(tuple, sn_coords))

                    shared_idx1 = [i for i, coord in enumerate(coords_tuples) if coord in sn_coords_tuples]
                    
                    coords = coords[shared_idx1]
                    feature = feature[shared_idx1]

                    shared_coords = list(map(tuple, coords))  # already filtered above
                    shared_idx2 = [i for i, coord in enumerate(sn_coords_tuples) if coord in shared_coords]
                    sn_index_dict = {tuple(c): i for i, c in enumerate(sn_coords)}
                    shared_idx2 = [sn_index_dict[coord] for coord in shared_coords]

                    sn_coords = sn_coords[shared_idx2]
                    sn_feature = sn_feature[shared_idx2]
                    
                    print(shared_idx1[:10], shared_idx2[:10], coords==sn_coords)
                    print(coords.shape, feature.shape, sn_coords.shape, sn_feature.shape)

                    sn_whole_mean_feature = torch.mean(sn_feature, axis=0)
                    torch.save(sn_feature, os.path.join(sn_whole_cancer_total_feature_save_dir, slide_id + '.pt'))
                    torch.save(sn_whole_mean_feature, os.path.join(sn_whole_cancer_mean_feature_save_dir, slide_id + '.pt'))
                    np.save(os.path.join(sn_whole_cancer_coords_save_dir, slide_id + '.npy'), sn_coords)
                    print(cancer_feature.shape, non_cancer_feature.shape, feature.shape, sn_cancer_feature.shape, sn_non_cancer_feature.shape, sn_feature.shape)
                    print(mean_feature.shape, non_mean_feature.shape, whole_mean_feature.shape, sn_mean_feature.shape, sn_non_mean_feature.shape, sn_whole_mean_feature.shape)
                
                if cancer_only:
                    if cancer_only:
                        xml_root_dir = os.path.join(xml_root_dir, database)
                        xml_subtype_dir = os.path.join(xml_root_dir, subtype_name)
                    annotation = slide_id + '.xml'
                    tree = ET.parse(os.path.join(xml_subtype_dir, annotation))
                    root = tree.getroot()

                    final_index_list = []
                    final_non_index_list = []
                    for region in root.findall('.//Region'):
                        vertices = region.findall('.//Vertex')
                        x = [float(vertex.get('X')) for vertex in vertices]
                        y = [float(vertex.get('Y')) for vertex in vertices]
                        polygon = Polygon(zip(x, y))
                        for idx, (coord_x, coord_y) in enumerate(coords):
                            point = Point(coord_x, coord_y)
                            if polygon.contains(point):
                                final_index_list.append(idx)
                    final_non_index_list = [idx for idx in range(len(coords)) if idx not in final_index_list]
                    if len(final_index_list) != 0:
                        cancer_feature = feature[final_index_list]
                        cancer_coords = coords[final_index_list]
                        mean_feature = torch.mean(cancer_feature, axis = 0)
                        torch.save(cancer_feature, os.path.join(cancer_only_total_feature_save_dir, slide_id + '.pt'))
                        torch.save(mean_feature, os.path.join(cancer_only_mean_feature_save_dir, slide_id + '.pt'))
                        np.save(os.path.join(cancer_only_coords_save_dir, slide_id + '.npy'), cancer_coords)

                    if len(final_non_index_list) != 0:
                        non_cancer_feature = feature[final_non_index_list]
                        non_cancer_coords = coords[final_non_index_list]
                        non_mean_feature = torch.mean(non_cancer_feature, axis = 0)
                        torch.save(non_cancer_feature, os.path.join(non_cancer_total_feature_save_dir, slide_id + '.pt'))
                        torch.save(non_mean_feature, os.path.join(non_cancer_mean_feature_save_dir, slide_id + '.pt'))
                        np.save(os.path.join(non_cancer_coords_save_dir, slide_id + '.npy'), non_cancer_coords)
                    
                    if extract_stainnorm:
                        if len(final_index_list) != 0:
                            sn_cancer_feature = sn_feature[final_index_list]
                            sn_cancer_coords = sn_coords[final_index_list]
                            sn_mean_feature = torch.mean(sn_cancer_feature, axis = 0)
                            torch.save(sn_cancer_feature, os.path.join(sn_cancer_only_total_feature_save_dir, slide_id + '.pt'))
                            torch.save(sn_mean_feature, os.path.join(sn_cancer_only_mean_feature_save_dir, slide_id + '.pt'))
                            np.save(os.path.join(sn_cancer_only_coords_save_dir, slide_id + '.npy'), sn_cancer_coords)
                        if len(final_non_index_list) != 0:
                            sn_non_cancer_feature = sn_feature[final_non_index_list]
                            sn_non_cancer_coords = sn_coords[final_non_index_list]
                            sn_non_mean_feature = torch.mean(sn_non_cancer_feature, axis = 0)
                            torch.save(sn_non_cancer_feature, os.path.join(sn_non_cancer_total_feature_save_dir, slide_id + '.pt'))
                            torch.save(sn_non_mean_feature, os.path.join(sn_non_cancer_mean_feature_save_dir, slide_id + '.pt'))
                            np.save(os.path.join(sn_non_cancer_coords_save_dir, slide_id + '.npy'), sn_non_cancer_coords)
                    
                pbar.update()

def Parser_main():
    parser = argparse.ArgumentParser(description="Extract feature")
    parser.add_argument("--savedir", help = 'Directory to save the feature',type = str, required = True)
    parser.add_argument("--featuredir", help = 'Directory where the trident feature saved', type = str, required = True)
    parser.add_argument("--metadata", help = 'metadata csv file path', type = str, required = True)
    parser.add_argument("--xml_root_dir", default = None, help = 'Directory where the xml file exist(if canceronly == True)', type = str)
    parser.add_argument("--pfm_list", nargs = "+", default = [], help = 'list of PFMs', type = str)
    parser.add_argument("--extract_stainnorm" , default = False, help = 'Extract stainnorm data, stainnorm data name is {pfm}_stainnorm', type = bool)
    parser.add_argument("--cancer_only", default = False, help = 'If you want to restrict the feature for canceronly (Need annotation_file)')
    return parser.parse_args()


def main():
    Argument = Parser_main()
    extract_feature(Argument.savedir, Argument.featuredir, Argument.metadata, Argument.pfm_list, Argument.xml_root_dir,  Argument.extract_stainnorm)

        
if __name__ == "__main__":
    main()