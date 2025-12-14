import os
import shutil
import torch
import h5py
from tqdm import tqdm
import numpy as np
import pandas as pd

# target_feature_list = ['virchow', 'virchow2', 'UNI','UNI2', 'ProvGiga', 'CONCH']
# patch_size_list = ['224', '224', '256', '256', '256', '512']

target_feature_list = ['virchow']
patch_size_list = ['224']

for target_feature, patch_size in zip(target_feature_list, patch_size_list):
    if target_feature == 'UNI':
        target_feature = 'uni_v1'
    elif target_feature == 'UNI2':
        target_feature = 'uni_v2'
    elif target_feature == 'ProvGiga':
        target_feature = 'gigapath'
    elif target_feature == 'CONCH':
        target_feature = 'conch_v1'

    magnification = '20x'
    # database_list = ['BORAMAE']
    # database_list = ['TCGA', 'KHMC', 'BORAMAE', 'GSH', 'GSH_histech', 'SNUBH']
    # database_list = ['GSYUHS', 'GSYUHS_histech']
    # database_list = ['GSH', 'GSH_histech']
    # database_list = ['SNUBH']
    database_list = ['BORAMAE']

    extract_stainnorm = True
    slide_level = False

    center_dict = {'KIRP': {'BQ': 'MSKCC', 'DW': 'NCI Urologic Oncology Branch', 'P4': 'MD Anderson Cancer Center'},
                'KIRC': {'BP': 'MSKCC', 'DV': 'NCI Urologic Oncology Branch', 'CJ': 'MD Anderson Cancer Center'},
                'KICH': {'KL': 'MSKCC', 'KM': 'NCI Urologic Oncology Branch', 'KO': 'MD Anderson Cancer Center'}}
    train_centers = ['MSKCC', 'NCI Urologic Oncology Branch', 'MD Anderson Cancer Center']
    # subtype_list = ['KIRP', 'KICH', 'KIRC']
    subtype_list = ['KICH']

    whole_meta = '/mnt/disk1/Kidney/Submission_dir/metadata.csv'
    whole_meta = pd.read_csv(whole_meta)
    whole_meta = whole_meta[whole_meta['subtype'].isin(subtype_list)]

    os.makedirs("/mnt/disk1/Kidney/Submission_dir/dataset/20x/Inference/", exist_ok= True)
    for database in database_list:
        root_dir = "/mnt/disk1/Kidney/Submission_dir/dataset/20x/Inference/%s" % database
        if database in ['TCGA']:
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
            elif database == 'Dartmouth':
                meta_file = '/mnt/disk1/Kidney/Final_analysis/Kidney_metadata.csv'
                meta_df = pd.read_csv(meta_file)
                meta_df = meta_df.dropna()

                meta_df['id'] = meta_df['sample']
                meta_df['subtype'] = meta_df['correct_subtype']
            
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
            
            
        feature_dir = "/mnt/disk2/trident_processing"

        # tumor_subtype_list = ['KIRP', 'KICH', 'KIRC']
        tumor_subtype_list = ['KICH']
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
            
            if slide_level:
                if target_feature == 'PRISM':
                    base_feature = 'virchow'
                else:
                    base_feature = target_feature
                subtype_feature_dir = os.path.join(feature_dir, database, subtype_name, magnification, patch_size, base_feature, 'slide_feature')
            else:
                subtype_feature_dir = os.path.join(feature_dir, database, subtype_name, magnification + '_' + patch_size + 'px_0px_overlap', 'features_' + target_feature)
                if extract_stainnorm:
                    sn_subtype_feature_dir = os.path.join(feature_dir, database, subtype_name, magnification + '_' + patch_size + 'px_0px_overlap', 'features_' + target_feature + '_stainnorm')
                print(database, subtype_feature_dir, tumor_subtype_dir)
            if slide_level:
                feature_save_dir = os.path.join(tumor_subtype_dir, 'slide_feature')
                if os.path.exists(feature_save_dir) is False:
                    os.mkdir(feature_save_dir)
            else:
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
                            subtype_feature_dir = os.path.join(feature_dir, database, 'KIRP', magnification, patch_size, base_feature, 'slide_feature')
                            if extract_stainnorm:
                                sn_subtype_feature_dir = os.path.join(feature_dir, database, 'KIRP', magnification, patch_size, target_feature+'_stainnorm', 'TCGA-B0-4827-01Z-00-DX1.c08eeafe-d2d4-41d1-a8f5-eacd1b0e601f', 'feature', 'pt_files')
                                
                        feature_file = os.path.join(subtype_feature_dir, slide_id + '.pt')
                        shutil.copy(feature_file, os.path.join(feature_save_dir, slide_id + '.pt'))
                        pbar.update()
                    
                else:
                    for f in target_files:
                        if f == 'BS21-24393-2.h5':
                            subtype_feature_dir = os.path.join(feature_dir, database, subtype_name, magnification + '_' + patch_size + 'px_0px_overlap', 'features_' + target_feature)
                            if extract_stainnorm:
                                sn_subtype_feature_dir = os.path.join(feature_dir, database, subtype_name, magnification + '_' + patch_size + 'px_0px_overlap', 'features_' + target_feature + '_stainnorm')
                        
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


    if False:
        rootdir = "/mnt/disk1/Kidney/patch_level_analysis/Submission/dataset/20x/Inference"

        total_dict = {k: [] for k in ['model', 'center', 'subtype', 'n']}
        for t in os.listdir(rootdir):
            if t == 'stat.csv':
                continue
            for m in os.listdir(os.path.join(rootdir, t)):
                for d in os.listdir(os.path.join(rootdir, t, m)):
                    for s in os.listdir(os.path.join(rootdir, t, m, d)):
                        total_dict['model'].append(m)
                        total_dict['center'].append(t)
                        total_dict['subtype'].append(s)
                        total_dict['n'].append(len(os.listdir(os.path.join(rootdir, t, m, d, s, 'coords'))))
                            
                            
        total_df = pd.DataFrame(total_dict)
        total_df.to_csv(os.path.join(rootdir, 'stat.csv'), index=False)
        print(total_df)