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

center_dict = {'KIRP': {'BQ': 'MSKCC', 'DW': 'NCI Urologic Oncology Branch', 'P4': 'MD Anderson Cancer Center'},
            'KIRC': {'BP': 'MSKCC', 'DV': 'NCI Urologic Oncology Branch', 'CJ': 'MD Anderson Cancer Center'},
            'KICH': {'KL': 'MSKCC', 'KM': 'NCI Urologic Oncology Branch', 'KO': 'MD Anderson Cancer Center'}}

tumor_subtype_list = ['KIRC', 'KIRP', 'KICH']

patch_dict = {'virchow': '224', 'virchow2': '224', 'UNI': '256', 'UNI2': '256', 'gigapath': '256', 'CONCH': '512'}

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

def extract_prototype_feature(save_dir, feature_dir, database, xml_root_dir, pfm_list, extract_stainnorm = True):
    save_dir = os.path.join(save_dir, 'Prototype')
    os.makedirs(save_dir, exist_ok = True)

    whole_cancer_save_dir = os.path.join(save_dir, 'whole_cancer')
    cancer_only_save_dir = os.path.join(save_dir, 'cancer_only')
    non_cancer_save_dir = os.path.join(save_dir, 'non_cancer')

    os.makedirs(whole_cancer_save_dir, exist_ok = True)
    os.makedirs(cancer_only_save_dir, exist_ok = True)
    os.makedirs(non_cancer_save_dir, exist_ok = True)

    xml_root_dir = os.path.join(xml_root_dir, database)
    for pfm in pfm_list:
        patch_size = patch_dict[pfm]
        if pfm == 'UNI':
            pfm = 'uni_v1'
        elif pfm == 'UNI2':
            pfm = 'uni_v2'
        elif pfm == 'CONCH':
            pfm = 'conch_v1'

        extract_stainnorm = True

        whole_cancer_target_feature_dir = os.path.join(whole_cancer_save_dir, pfm)
        cancer_only_target_feature_dir = os.path.join(cancer_only_save_dir, pfm)
        non_cancer_target_feature_dir = os.path.join(non_cancer_save_dir, pfm)

        os.makedirs(whole_cancer_target_feature_dir, exist_ok = True)
        os.makedirs(cancer_only_target_feature_dir, exist_ok = True)
        os.makedirs(non_cancer_target_feature_dir, exist_ok = True)

        whole_cancer_patch_dir = os.path.join(whole_cancer_target_feature_dir, patch_size)
        cancer_only_patch_dir = os.path.join(cancer_only_target_feature_dir, patch_size)
        non_cancer_patch_dir = os.path.join(non_cancer_target_feature_dir, patch_size)
        
        os.makedirs(whole_cancer_patch_dir, exist_ok = True)
        os.makedirs(cancer_only_patch_dir, exist_ok = True)
        os.makedirs(non_cancer_patch_dir, exist_ok = True)

        if extract_stainnorm:
            sn_whole_cancer_target_feature_dir = os.path.join(whole_cancer_save_dir, pfm + '_stainnorm')
            sn_cancer_only_target_feature_dir = os.path.join(cancer_only_save_dir, pfm + '_stainnorm')
            sn_non_cancer_target_feature_dir = os.path.join(non_cancer_save_dir, pfm + '_stainnorm')

            os.makedirs(sn_whole_cancer_target_feature_dir, exist_ok = True)
            os.makedirs(sn_cancer_only_target_feature_dir, exist_ok = True)
            os.makedirs(sn_non_cancer_target_feature_dir, exist_ok = True)

            sn_whole_cancer_patch_dir = os.path.join(sn_whole_cancer_target_feature_dir, patch_size)
            sn_cancer_only_patch_dir = os.path.join(sn_cancer_only_target_feature_dir, patch_size)
            sn_non_cancer_patch_dir = os.path.join(sn_non_cancer_target_feature_dir, patch_size)
            
            os.makedirs(sn_whole_cancer_patch_dir, exist_ok = True)
            os.makedirs(sn_cancer_only_patch_dir, exist_ok = True)
            os.makedirs(sn_non_cancer_patch_dir, exist_ok = True)
            

        for tumor_subtype in tumor_subtype_list:
            subtype_name = tumor_subtype
            whole_cancer_subtype_dir = os.path.join(whole_cancer_patch_dir, tumor_subtype)
            cancer_only_subtype_dir = os.path.join(cancer_only_patch_dir, tumor_subtype)
            non_cancer_subtype_dir = os.path.join(non_cancer_patch_dir, tumor_subtype)

            os.makedirs(whole_cancer_subtype_dir, exist_ok = True)
            os.makedirs(cancer_only_subtype_dir, exist_ok = True)
            os.makedirs(non_cancer_subtype_dir, exist_ok = True)

            subtype_feature_dir = os.path.join(feature_dir, database, subtype_name, '20x_' + str(patch_size) + 'px_0px_overlap', 'features_' + pfm)

            whole_cancer_total_feature_save_dir = os.path.join(whole_cancer_subtype_dir, 'total_feature')
            whole_cancer_mean_feature_save_dir = os.path.join(whole_cancer_subtype_dir, 'mean_feature')
            whole_cancer_coords_save_dir = os.path.join(whole_cancer_subtype_dir, 'coords')

            cancer_only_total_feature_save_dir = os.path.join(cancer_only_subtype_dir, 'total_feature')
            cancer_only_mean_feature_save_dir = os.path.join(cancer_only_subtype_dir, 'mean_feature')
            cancer_only_coords_save_dir = os.path.join(cancer_only_subtype_dir, 'coords')
            
            non_cancer_total_feature_save_dir = os.path.join(non_cancer_subtype_dir, 'total_feature')
            non_cancer_mean_feature_save_dir = os.path.join(non_cancer_subtype_dir, 'mean_feature')
            non_cancer_coords_save_dir = os.path.join(non_cancer_subtype_dir, 'coords')
            
            os.makedirs(whole_cancer_total_feature_save_dir, exist_ok = True)
            os.makedirs(whole_cancer_mean_feature_save_dir, exist_ok = True)
            os.makedirs(whole_cancer_coords_save_dir, exist_ok = True)

            os.makedirs(cancer_only_total_feature_save_dir, exist_ok = True)
            os.makedirs(cancer_only_mean_feature_save_dir, exist_ok = True)
            os.makedirs(cancer_only_coords_save_dir, exist_ok = True)

            os.makedirs(non_cancer_total_feature_save_dir, exist_ok = True)
            os.makedirs(non_cancer_mean_feature_save_dir, exist_ok = True)
            os.makedirs(non_cancer_coords_save_dir, exist_ok = True)
            
            if extract_stainnorm:
                sn_subtype_feature_dir = os.path.join(feature_dir, database, subtype_name, '20x_' + str(patch_size) + 'px_0px_overlap', 'features_' + pfm + '_stainnorm')
                
                sn_whole_cancer_subtype_dir = os.path.join(sn_whole_cancer_patch_dir, tumor_subtype)
                sn_cancer_only_subtype_dir = os.path.join(sn_cancer_only_patch_dir, tumor_subtype)
                sn_non_cancer_subtype_dir = os.path.join(sn_non_cancer_patch_dir, tumor_subtype)

                os.makedirs(sn_whole_cancer_subtype_dir, exist_ok = True)
                os.makedirs(sn_cancer_only_subtype_dir, exist_ok = True)
                os.makedirs(sn_non_cancer_subtype_dir, exist_ok = True)

                sn_whole_cancer_total_feature_save_dir = os.path.join(sn_whole_cancer_subtype_dir, 'total_feature')
                sn_whole_cancer_mean_feature_save_dir = os.path.join(sn_whole_cancer_subtype_dir, 'mean_feature')
                sn_whole_cancer_coords_save_dir = os.path.join(sn_whole_cancer_subtype_dir, 'coords')

                sn_cancer_only_total_feature_save_dir = os.path.join(sn_cancer_only_subtype_dir, 'total_feature')
                sn_cancer_only_mean_feature_save_dir = os.path.join(sn_cancer_only_subtype_dir, 'mean_feature')
                sn_cancer_only_coords_save_dir = os.path.join(sn_cancer_only_subtype_dir, 'coords')
                
                sn_non_cancer_total_feature_save_dir = os.path.join(sn_non_cancer_subtype_dir, 'total_feature')
                sn_non_cancer_mean_feature_save_dir = os.path.join(sn_non_cancer_subtype_dir, 'mean_feature')
                sn_non_cancer_coords_save_dir = os.path.join(sn_non_cancer_subtype_dir, 'coords')
                
                os.makedirs(sn_whole_cancer_total_feature_save_dir, exist_ok = True)
                os.makedirs(sn_whole_cancer_mean_feature_save_dir, exist_ok = True)
                os.makedirs(sn_whole_cancer_coords_save_dir, exist_ok = True)

                os.makedirs(sn_cancer_only_total_feature_save_dir, exist_ok = True)
                os.makedirs(sn_cancer_only_mean_feature_save_dir, exist_ok = True)
                os.makedirs(sn_cancer_only_coords_save_dir, exist_ok = True)

                os.makedirs(sn_non_cancer_total_feature_save_dir, exist_ok = True)
                os.makedirs(sn_non_cancer_mean_feature_save_dir, exist_ok = True)
                os.makedirs(sn_non_cancer_coords_save_dir, exist_ok = True)
            
            xml_subtype_dir = os.path.join(xml_root_dir, tumor_subtype)
            if database == 'TCGA':
                annotation_file_list = [d for d in os.listdir(xml_subtype_dir) if (d.split('-')[1] in center_dict[tumor_subtype].keys())]

            with tqdm(total = len(annotation_file_list)) as pbar:
                for annotation in annotation_file_list:
                    if annotation.endswith('.xml'):
                        slide_id = annotation.split('.xml')[0]
                        tree = ET.parse(os.path.join(xml_subtype_dir, annotation))
                        root = tree.getroot()
                        try:
                            whole_data = h5py.File(os.path.join(subtype_feature_dir, slide_id + '.h5'), 'r')
                        except:
                            print(f'No data of {slide_id}')
                            continue

                        print(tumor_subtype, slide_id)
                        
                        feature = torch.tensor(whole_data['features'][()])
                        coords = whole_data['coords'][()]
                        
                        if extract_stainnorm:
                            try:
                                sn_data = h5py.File(os.path.join(sn_subtype_feature_dir, slide_id + '.h5'), 'r')
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
                            
                        final_index_list = []
                        final_non_index_list = []
                        for region in root.findall('.//Region'):
                            vertices = region.findall('.//Vertex')
                            x = [float(vertex.get('X')) for vertex in vertices]
                            y = [float(vertex.get('Y')) for vertex in vertices]
                            # Create a polygon from the region's vertices
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

                        whole_mean_feature = torch.mean(feature, axis=0)
                        torch.save(feature, os.path.join(whole_cancer_total_feature_save_dir, slide_id + '.pt'))
                        torch.save(whole_mean_feature, os.path.join(whole_cancer_mean_feature_save_dir, slide_id + '.pt'))
                        np.save(os.path.join(whole_cancer_coords_save_dir, slide_id + '.npy'), coords)
                        
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
                            
                            sn_whole_mean_feature = torch.mean(sn_feature, axis=0)
                            torch.save(sn_feature, os.path.join(sn_whole_cancer_total_feature_save_dir, slide_id + '.pt'))
                            torch.save(sn_whole_mean_feature, os.path.join(sn_whole_cancer_mean_feature_save_dir, slide_id + '.pt'))
                            np.save(os.path.join(sn_whole_cancer_coords_save_dir, slide_id + '.npy'), sn_coords)
                            print(cancer_feature.shape, non_cancer_feature.shape, feature.shape, sn_cancer_feature.shape, sn_non_cancer_feature.shape, sn_feature.shape)
                            print(mean_feature.shape, non_mean_feature.shape, whole_mean_feature.shape, sn_mean_feature.shape, sn_non_mean_feature.shape, sn_whole_mean_feature.shape)
                        
                    pbar.update()

def extract_inference_feature(save_dir, pfm_list, database_list, extract_stainnorm, whole_meta, feature_dir):
    
    for pfm in pfm_list:
        patch_size = patch_dict[pfm]

        if target_feature == 'UNI':
            target_feature = 'uni_v1'
        elif target_feature == 'UNI2':
            target_feature = 'uni_v2'
        elif target_feature == 'CONCH':
            target_feature = 'conch_v1'

        whole_meta = pd.read_csv(whole_meta)
        whole_meta = whole_meta[whole_meta['subtype'].isin(tumor_subtype_list)]

        save_dir = os.path.join(save_dir, 'Inference')
        os.makedirs(save_dir, exist_ok= True)
        for database in database_list:
            database_save_dir = os.path.join(save_dir, database)
            
            if database == 'TCGA':
                use_meta = False
            else:
                use_meta = True

            if use_meta:
                if database == 'KHMC':
                    meta_file = '/mnt/disk1/Kidney/patch_level_analysis/KHMC_meta.csv'
                    meta_df = pd.read_csv(meta_file)

                    meta_df['id'] = ['K'+id.split('-')[0]+'-'+id.split('-')[1].lstrip('0')+'-'+str(b) for id, b in zip(meta_df['id'].tolist(), meta_df['Block'].tolist())]

                    subtype = []
                    for i in range(len(meta_df)):
                        type = meta_df.iloc[i]['Type']
                        if type == 'P':
                            subtype.append('KIRP')
                        elif type == 'CH':
                            subtype.append('KICH')
                        else:
                            subtype.append('KIRC')
                    meta_df['subtype'] = subtype
                elif (database == 'GSH') or (database == 'GSH_histech'):
                    meta_file = '/mnt/disk1/Kidney/patch_level_analysis/GSYUHS_JHP_comment.csv'
                    meta_df = pd.read_csv(meta_file)

                    meta_df['id'] = [s.split('.')[0] for s in meta_df['sample'].tolist()]
                    subtype = []
                    for i in range(len(meta_df)):
                        type = meta_df.iloc[i]['JHP']
                        if type == 'PRCC':
                            subtype.append('KIRP')
                        elif type == 'CHRCC':
                            subtype.append('KICH')
                        else:
                            subtype.append('KIRC')
                    meta_df['subtype'] = subtype

                elif database == 'SNUBH':
                    meta_file = '/mnt/disk1/Kidney/patch_level_analysis/SNUBH_meta.csv'
                    meta_df = pd.read_csv(meta_file)
                    meta_df = meta_df.dropna()

                    meta_df['id'] = ['SNUBH_S'+s.split(' ')[-1].split('-')[0]+'-'+s.split('-')[-1].lstrip('0') for s in meta_df['sample']]
                    meta_df['subtype'] = meta_df['cancer']
                
                elif database == 'BORAMAE':
                    meta_file = '/mnt/disk1/Kidney/Submission_dir/metadata.csv'
                    meta_df = pd.read_csv(meta_file)
                    meta_df = meta_df.dropna()
                    meta_df = meta_df[meta_df['center'] == 'BORAMAE']
                    meta_df['id'] = meta_df['Patient']
                print(meta_df)


            target_feature_dir = os.path.join(root_dir, target_feature)
            patch_dir = os.path.join(target_feature_dir, patch_size)
            if os.path.exists(root_dir) is False:
                os.mkdir(root_dir)
            if os.path.exists(target_feature_dir) is False:
                os.mkdir(target_feature_dir)
            if os.path.exists(patch_dir) is False:
                os.mkdir(patch_dir)
            if extract_stainnorm:
                sn_target_feature_dir = os.path.join(root_dir, target_feature+'_stainnorm')
                sn_patch_dir = os.path.join(sn_target_feature_dir, patch_size)
                if os.path.exists(sn_target_feature_dir) is False:
                    os.mkdir(sn_target_feature_dir)
                if os.path.exists(sn_patch_dir) is False:
                    os.mkdir(sn_patch_dir)
                
            for tumor_subtype in tumor_subtype_list:
                subtype_name = tumor_subtype
                tumor_subtype_dir = os.path.join(patch_dir, tumor_subtype)
                if os.path.exists(tumor_subtype_dir) is False:
                    os.mkdir(tumor_subtype_dir)
                if extract_stainnorm:
                    sn_tumor_subtype_dir = os.path.join(sn_patch_dir, tumor_subtype)
                    if os.path.exists(sn_tumor_subtype_dir) is False:
                        os.mkdir(sn_tumor_subtype_dir)
                        
                if database not in ['TCGA']:
                    subtype_name = 'RCC'
                
        
                subtype_feature_dir = os.path.join(feature_dir, database, subtype_name, '20x' + '_' + patch_size + 'px_0px_overlap', 'features_' + target_feature)
                if extract_stainnorm:
                    sn_subtype_feature_dir = os.path.join(feature_dir, database, subtype_name, '20x' + '_' + patch_size + 'px_0px_overlap', 'features_' + target_feature + '_stainnorm')
                print(database, subtype_feature_dir, tumor_subtype_dir)
                
                
                total_feature_save_dir = os.path.join(tumor_subtype_dir, 'total_feature')
                mean_feature_save_dir = os.path.join(tumor_subtype_dir, 'mean_feature')
                coords_save_dir = os.path.join(tumor_subtype_dir, 'coords')
                if os.path.exists(total_feature_save_dir) is False:
                    os.mkdir(total_feature_save_dir)
                if os.path.exists(mean_feature_save_dir) is False:
                    os.mkdir(mean_feature_save_dir)
                if os.path.exists(coords_save_dir) is False:
                    os.mkdir(coords_save_dir)
                if extract_stainnorm:
                    sn_total_feature_save_dir = os.path.join(sn_tumor_subtype_dir, 'total_feature')
                    sn_mean_feature_save_dir = os.path.join(sn_tumor_subtype_dir, 'mean_feature')
                    sn_coords_save_dir = os.path.join(sn_tumor_subtype_dir, 'coords')
                    if os.path.exists(sn_total_feature_save_dir) is False:
                        os.mkdir(sn_total_feature_save_dir)
                    if os.path.exists(sn_mean_feature_save_dir) is False:
                        os.mkdir(sn_mean_feature_save_dir)
                    if os.path.exists(sn_coords_save_dir) is False:
                        os.mkdir(sn_coords_save_dir)
                    
                if use_meta:
                    if database == 'SNUBH':
                        target_files = [d for d in os.listdir(subtype_feature_dir) if '-'.join(d.split('.')[0].split('-')[:-1]) in meta_df[meta_df['subtype'] == tumor_subtype]['id'].tolist()]
                        print([d for d in os.listdir(subtype_feature_dir) if '-'.join(d.split('.')[0].split('-')[:-1]) not in meta_df['id'].tolist()])
                    elif database == 'BORAMAE':
                        target_files = [d for d in os.listdir(subtype_feature_dir) if d not in ['BS22-21615-3.h5', 'BS20-8613-6.h5', 'BS21-24393-2.h5']]
                        target_files = [d for d in target_files if d.split('.')[0] in meta_df[meta_df['subtype'] == tumor_subtype]['id'].tolist()]
                        if tumor_subtype == 'KIRC':
                            target_files.append('BS21-24393-2.h5')
                    else:
                        target_files = [d for d in os.listdir(subtype_feature_dir) if d.split('.')[0] in meta_df[meta_df['subtype'] == tumor_subtype]['id'].tolist()]
                elif database == 'TCGA':
                    ## slide-level training set
                    # target_files = [d for d in os.listdir(subtype_feature_dir) if d in os.listdir('/mnt/disk1/Kidney/patch_level_analysis/dataset/20x/whole_cancer/ProvGiga/256/%s/total_feature/' % tumor_subtype)]
                    # target_files = [d for d in os.listdir(subtype_feature_dir) if d.split('-')[1] in center_dict[tumor_subtype].keys()]
                    target_files = [d for d in os.listdir(subtype_feature_dir) if d.split('-')[1] not in center_dict[tumor_subtype].keys()]
                    target_files = [d for d in target_files if ('11Z' not in d) and (d not in ['TCGA-UW-A7GJ-01Z', 'TCGA-5P-A9KA-01Z', 'TCGA-UZ-A9PQ-01Z'])]
                
                if database != 'SNUBH':
                    target_files = [f for f in target_files if '-'.join(f.split('.')[0].split('-')[:3]) in whole_meta['Patient'].tolist()]
                target_files = ['BS21-6333-1.h5']
                with tqdm(total = len(target_files)) as pbar:
                    if slide_level:
                        for f in target_files:
                            slide_id = f.split('.h5')[0]
                            
                            if f == 'BS21-24393-2.h5':
                                subtype_feature_dir = os.path.join(feature_dir, database, 'KIRP', '20x', patch_size, base_feature, 'slide_feature')
                                if extract_stainnorm:
                                    sn_subtype_feature_dir = os.path.join(feature_dir, database, 'KIRP', '20x', patch_size, target_feature+'_stainnorm', 'TCGA-B0-4827-01Z-00-DX1.c08eeafe-d2d4-41d1-a8f5-eacd1b0e601f', 'feature', 'pt_files')
                                    
                            feature_file = os.path.join(subtype_feature_dir, slide_id + '.pt')
                            shutil.copy(feature_file, os.path.join(feature_save_dir, slide_id + '.pt'))
                            pbar.update()
                        
                    else:
                        for f in target_files:
                            if f == 'BS21-24393-2.h5':
                                subtype_feature_dir = os.path.join(feature_dir, database, subtype_name, '20x_' + patch_size + 'px_0px_overlap', 'features_' + target_feature)
                                if extract_stainnorm:
                                    sn_subtype_feature_dir = os.path.join(feature_dir, database, subtype_name, '20x_' + patch_size + 'px_0px_overlap', 'features_' + target_feature + '_stainnorm')
                            
                            slide_id = f.split('.h5')[0]
                            if os.path.exists(os.path.join(mean_feature_save_dir, slide_id + '.h5')):
                                pbar.update()
                                continue
                            
                            feature_file = os.path.join(subtype_feature_dir, slide_id + '.h5')
                            whole_feature = h5py.File(feature_file, 'r')
                            feature = torch.tensor(whole_feature['features'][()])
                            coords = whole_feature['coords'][()]
                            
                            if extract_stainnorm:
                                sn_feature_file = os.path.join(sn_subtype_feature_dir, slide_id + '.h5')
                                sn_data = h5py.File(sn_feature_file, 'r')
                                sn_feature = torch.tensor(sn_data['features'][()])
                                sn_coords = sn_data['coords'][()]
                                
                                coords_tuples = list(map(tuple, coords))
                                sn_coords_tuples = list(map(tuple, sn_coords))

                                shared_idx1 = [i for i, coord in enumerate(coords_tuples) if coord in sn_coords_tuples]
                                
                                coords = coords[shared_idx1]
                                feature = feature[shared_idx1]

                                shared_coords = list(map(tuple, coords))  # already filtered above
                                sn_index_map = {coord: i for i, coord in enumerate(sn_coords_tuples)}
                                shared_idx2 = [i for i, coord in enumerate(sn_coords_tuples) if coord in shared_coords]
                                sn_index_dict = {tuple(c): i for i, c in enumerate(sn_coords)}
                                shared_idx2 = [sn_index_dict[coord] for coord in shared_coords]

                                sn_coords = sn_coords[shared_idx2]
                                sn_feature = sn_feature[shared_idx2]
                                
                                print(shared_idx1[:10], shared_idx2[:10], coords==sn_coords)
                                print(coords.shape, feature.shape, sn_coords.shape, sn_feature.shape)
                                
                            mean_feature = torch.mean(feature, axis = 0)
                            torch.save(feature, os.path.join(total_feature_save_dir, slide_id + '.pt'))
                            torch.save(mean_feature, os.path.join(mean_feature_save_dir, slide_id + '.pt'))
                            np.save(os.path.join(coords_save_dir, slide_id + '.npy'), coords)
                            
                            if extract_stainnorm:
                                sn_mean_feature = torch.mean(sn_feature, axis = 0)
                                torch.save(sn_feature, os.path.join(sn_total_feature_save_dir, slide_id + '.pt'))
                                torch.save(sn_mean_feature, os.path.join(sn_mean_feature_save_dir, slide_id + '.pt'))
                                np.save(os.path.join(sn_coords_save_dir, slide_id + '.npy'), sn_coords)
                            
                            pbar.update()

def Parser_main():
    parser = argparse.ArgumentParser(description="Extract feature for prototyping")
    parser.add_argument("--mode", default="Train", help="Train / Analysis / External_validation", type=str)
    parser.add_argument("--savedir", default = '/mnt/disk1/Kidney/Submission_dir/dataset/20x/', help = 'Directory to save the feature',type = str)
    parser.add_argument("--featuredir", default = "/mnt/disk2/trident_processing", help = 'Directory where the trident feature saved', type = str)
    parser.add_argument("--database", nargs = "+", default = ['TCGA', 'BORAMAE', 'KHMC', 'GSH', 'GSH_histech', 'SNUBH'], help = 'TCGA or Internal center, maybe another database can be used if you have', type = str)
    parser.add_argument("--xml_root_dir", default = '/mnt/disk1/Kidney/Submission_dir/Annotation/', help = 'Directory where the xml file exist', type = str)
    parser.add_argument("--pfm_list", nargs = "+", default = ['virchow', 'virchow2', 'UNI','UNI2', 'GigaPath', 'CONCH'], help = 'list of PFMs', type = int)
    parser.add_argument("--extract_stainnorm" , default = True, help = 'Extract stainnorm data, stainnorm data name is {pfm}_stainnorm', type = bool)
    parser.add_argument("--metadata", default = '/mnt/disk1/Kidney/Submission_dir/metadata.csv', help = 'metadata csv file path', type = str)
    return parser.parse_args()


def main():
    Argument = Parser_main()
    if Argument.mode == 'Prototype':
        extract_prototype_feature(Argument.savedir, Argument.featuredir, Argument.database, Argument.xml_root_dir, Argument.pfm_list, Argument.extract_stainnorm)

        
if __name__ == "__main__":
    main()