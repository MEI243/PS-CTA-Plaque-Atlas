import numpy as np
from pathlib import Path
import json

# import scipy.stats as stats
import matplotlib.pyplot as plt
# import seaborn as sns
import random

from matplotlib.colors import LinearSegmentedColormap

import pandas as pd
from tqdm import tqdm
import os
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils import centerline_length, centerline_length2, convert
from scipy.spatial.distance import cdist

from scipy.interpolate import splprep, splev
# from skimage.measure import label, regionprops
# from Analysis4 import Analysis4

AHA_segments = ['RCA', 'R-PLB', 'R-PDA', 'LM', 'LAD', 'LCX', 'RAMUS', 'OM1', 'OM2', 'D1', 'D2', 'L-PLB', 'L-PDA']
AHA18_segments = ['R-PLB', 'R-PDA', 'LM', 'RAMUS', 'OM1', 'OM2', 'D1', 'D2', 'L-PLB', 'L-PDA', \
                'RCA-Proximal', 'RCA-Middle', 'RCA-Distal', 'LAD-Proximal', 'LAD-Middle', 'LAD-Distal', 'LCX-Proximal', 'LCX-Distal']


class Averaging_Ctl():
    def __init__(self):
        super().__init__()
        self.root = './sample_data/'
        # dataset_size = 16300 # cohort size
        self.cohort_info = pd.read_csv(self.root+'cohort.csv')

        self.scpr_ctl_path = self.root+'centerline_internal/'

        self.ctl_root = Path(self.root)/'average_shape_results/centerline_internal_resampled2origin/'
        self.mean_shape_savep = Path(self.root)/'average_shape_results/'/'mean_segment_shapes.json'
        self.mean_concatpts_savep = Path(self.root)/'average_shape_results/'/'mean_concat_pts.json'
        self.segments = ['RCA', 'R-PLB', 'R-PDA', 'LAD', 'LCX', 'RAMUS', 'OM1', 'OM2', 'D1', 'D2', 'L-PLB', 'L-PDA']
        self.SEG_TREE = {'LM':['LAD', 'LCX', 'RAMUS'],
                    'LAD-1':['D1'],
                    'LAD-2':['D2'],
                    'LCX-1':['OM1'],
                    'LCX-2':['OM2'],
                    'LCX-3':['L-PLB'],
                    'LCX-4':['L-PDA'],
                    'RCA-1':['R-PLB'],
                    'RCA-2':['R-PDA'],
                    }
        self.colors = [
                "#FF7F0E",  
                "#1F77B4",  
                "#2CA02C",  
                "#FFD700",  
                "#9467BD",  
                "#8C564B",  
                "#E377C2",  
                "#A7EAEA",  
                "#BCBD22",  
                "#17BECF",  
                "#AEC7E8",  
                "#FF9896",  
                "#98DF8A",  
                "#FFBB78",  
                "#C5B0D5",  
                "#C49C94",  
                "#F7B6D2",  
                "#C7C7C7"   
            ]


        self.color_dict = {}
        for i, k in enumerate(self.segments):
            self.color_dict.update({k:self.colors[i]})
    
    def ctl_resampling(self): 

        '''resampling centerlines & align to origin'''
        'fix the number of points of each segment as the average number of points of each segment.'
        cp_cohort_info = copy.deepcopy(self.cohort_info)

        color_dict = {}
        for i, k in enumerate(self.segments):
            color_dict.update({k:self.colors[i]})
        
        # --get the mean of each coloumn
        sample_info=Path(self.root)/'statistics/mean_number_of_points.json'
        with open(sample_info) as f:
            mean = json.load(f)

        for index, row in tqdm(cp_cohort_info.iterrows(), total=len(cp_cohort_info)):
            resample_points = {}
            resample_origin_points = {}

            subj_name = Path(row['image_name']).stem
            # sample_row = sample_info[sample_info['image_name']==subj_name]
            origin = eval(cp_cohort_info['Origin'].values[0])
            for q in Path(Path(self.scpr_ctl_path)/subj_name).rglob('*.csv'):
                
                if 'Unnamed: 0' in pd.read_csv(q).columns:
                    segment_ctl = pd.read_csv(q, index_col=0).values.tolist()
                else:
                    segment_ctl = pd.read_csv(q).values.tolist()
                
                assert np.array(segment_ctl).shape[1]==3, "Wrong point dimension!"
                
                if len(segment_ctl)<5:
                    continue

                seg_name = q.stem
                if seg_name in ['RCA-Proximal', 'LAD-Proximal', 'LCX-Proximal']:
                    seg_name = seg_name.split('-')[0]

                if seg_name in self.segments:

                    target_num = mean[seg_name+'_num']
                    new_ctl = self.upsample_points_3d(np.array(segment_ctl), target_num)
                    resample_points.update({seg_name:new_ctl.tolist()})
                    
                    to_orig_ctl = np.array(new_ctl)-np.array(origin)
                    resample_origin_points.update({seg_name:to_orig_ctl.tolist()})

            for mb in ['LAD', 'LCX', 'RCA']:
                if mb not in resample_points.keys():
                    print(f'{subj_name} has incomplete main branches')
            

            (self.ctl_root).mkdir(parents=True, exist_ok=True)
            json2 = json.dumps(resample_origin_points, indent=4, ensure_ascii=False)
            with open(Path(self.ctl_root)/f'{subj_name}.json', 'w') as f:
                f.write(json2)
    
    def upsample_points_3d(self, points, C):
        """
        Upsample a set of 3D points using spline interpolation.

        :param points: An array of shape (n, 3) representing the original points.
        :param C: The desired number of points after upsampling.
        :return: An array of shape (C, 3) representing the upsampled points.
        """
        # if len(points) >= C:
        #     raise ValueError("Number of points to upsample must be greater than the original points")

        # Parameterize the points and fit a spline
        tck, u = splprep([points[:, 0], points[:, 1], points[:, 2]], s=0)
        
        # Generate new points on the spline
        u_new = np.linspace(0, 1, C)
        new_points = splev(u_new, tck)

        return np.vstack(new_points).T
    
    def preshape_mu_seg(self, ):
        
        mean_shape_seg = {}
        '1. get all intersection points'
        save_dict = {key:{'image':[], 'pointset':[]} for key in self.segments}
        all_intersec_points = {}
        for k1 in self.SEG_TREE:
            all_intersec_points.update({k1:{', '.join(self.SEG_TREE[k1]):[]}})
        subjs_index_intersec_points = copy.deepcopy(all_intersec_points)

        length_dict = {'image':[]}
        for key in self.segments:
            length_dict.update({f'{key}_length':[]})
        

        for i, subj_name in tqdm(enumerate(os.listdir(self.ctl_root)), total=len(os.listdir(self.ctl_root))): #[0:1000]
            # if i < 10238:
            #     continue
            with open(self.ctl_root/subj_name) as f:
                ctl_dict = json.load(f)
            with open(Path(self.root)/'intersection/'/subj_name) as f:
                intersection_dict = json.load(f)
            
            origin = eval(self.cohort_info['Origin'].values[0])
            length_dict['image'].append(Path(subj_name).stem)

            for seg_name in sorted(self.segments):
                if seg_name not in ctl_dict.keys():
                    length_dict[seg_name+'_length'].append(0)
                    continue
                else:
                    save_dict[seg_name]['pointset'].append(ctl_dict[seg_name])
                    save_dict[seg_name]['image'].append(Path(subj_name).stem)

                    length_dict[seg_name+'_length'].append(centerline_length(ctl_dict[seg_name]))
                    if seg_name in ['RCA', 'LAD', 'LCX']:
                        # find related keys
                        k1s = []
                        k2s = []
                        for k1 in all_intersec_points.keys():
                            if seg_name in k1:
                                k1s.append(k1)
                                k2s.append(list(all_intersec_points[k1].keys())[0])
                            
                            if (seg_name == 'LAD') and (seg_name in list(all_intersec_points[k1].keys())[0]):
                                k1s.append(k1)
                                k2s.append(list(all_intersec_points[k1].keys())[0])
                        
                        for j, key in enumerate(k1s):
                            if list(intersection_dict[key].keys())[0] != '':
                                to_orig_pt = np.array(intersection_dict[key][list(intersection_dict[key].keys())[0]])-np.array(origin)
                                all_intersec_points[key][k2s[j]].append(to_orig_pt.tolist())
                                subjs_index_intersec_points[key][k2s[j]].append(i)

        js1 = json.dumps(subjs_index_intersec_points, indent=4, ensure_ascii=False)
        with open(Path(self.root)/f'average_shape_results/subjs_index_intersec_points_sample.json', 'w') as f:
            f.write(js1)
        js2 = json.dumps(all_intersec_points, indent=4, ensure_ascii=False)
        with open(Path(self.root)/f'average_shape_results/intersection_points_all_sample.json', 'w') as f:
            f.write(js2)
        # pd.DataFrame(length_dict).to_csv(Path(self.root)/'average_shape_results/resampled2origin_v2_length.csv')
               
        '''
        The results of our cohort are given as 'mean_segment_shapes.json' and 'mean_concat_pts.json'        
        '''
        # mean of concate point
        mean_concate_pts = {}
        # mean of each segment
        for seg_name in sorted(self.segments):                
            if save_dict[seg_name]['pointset']==[]:
                continue
            
            js = json.dumps(save_dict[seg_name], indent=4, ensure_ascii=False)
            with open(Path(self.root)/f'average_shape_results/{seg_name}_points_all_sample.json', 'w') as f:
                f.write(js)

            pointset = np.array(save_dict[seg_name]['pointset']).transpose(2, 1, 0)
            std_pointsets, centroids = self.to_preshape(pointset, scale=False, return_center=True)
            mu_init = self.update_mean(std_pointsets, normalize=False)
            mean_shape_seg.update({seg_name:mu_init.T.tolist()})

            if seg_name in ['RCA', 'LAD', 'LCX']:
                # find related keys
                k1s = []
                k2s = []
                for k1 in all_intersec_points.keys():
                    if seg_name in k1:
                        k1s.append(k1)
                        k2s.append(list(all_intersec_points[k1].keys())[0])
                    
                    if (seg_name == 'LAD') and (seg_name in list(all_intersec_points[k1].keys())[0]):
                        k1s.append(k1)
                        k2s.append(list(all_intersec_points[k1].keys())[0])
            
                for j, key in enumerate(k1s):
                    # centroids should be the main branch centroids of subjects who have the intersection point.
                    ctr = centroids[:,:,subjs_index_intersec_points[key][list(all_intersec_points[key].keys())[0]]]

                    intersec_pts = all_intersec_points[key][list(all_intersec_points[key].keys())[0]]
                    if intersec_pts == []:
                        continue
                    std_intersec_pts = np.array(intersec_pts).transpose(1, 0)-np.squeeze(ctr)
                    intersec_mean_pt = np.mean(std_intersec_pts, axis=1)

                    # find closest point on mean_shape_seg[seg_name]
                    segments = [mean_shape_seg[seg_name], [intersec_mean_pt.tolist()]]
                    closest_pair, closest_pair_index, min_distance = self._find_closest_point_pair(segments)
                    concat_pt = mean_shape_seg[seg_name][closest_pair_index[0]]
                    mean_concate_pts.update({key:{list(all_intersec_points[key].keys())[0]:concat_pt}})
                    mean_concate_pts[key].update({'pt_index':float(closest_pair_index[0])})
            
        js = json.dumps(mean_shape_seg, indent=4, ensure_ascii=False)
        with open(self.mean_shape_savep.parent/'mean_segment_shapes_samples.json', 'w') as f:
            f.write(js)

        js3 = json.dumps(mean_concate_pts, indent=4, ensure_ascii=False)
        with open(self.mean_concatpts_savep.parent/'mean_concat_pts_samples.json', 'w') as f:
            f.write(js3)

        return mean_shape_seg
    
    def to_preshape(self, p, scale=True, return_center=False):
        """
        Description: Standardizes the pointset to pre-shape space
        
        Inputs:
            p   [DxN] or [DxNxM] : Pointset(s)
        
        Outputs:
            p   [DxN] or [DxNxM] : Standardized pointset(s)
        """

        # Assuming p is a NumPy array
        N = p.shape[1]
        
        # Shift by subtracting the centroid
        centroids = np.sum(p, axis=1, keepdims=True) / N
        p = p - centroids
        
        # Scale
        if scale:
            temp = p ** 2
            scale = np.sqrt(np.sum(temp, axis=(0, 1), keepdims=True))  # sums over D and N dimensions
            p = p / scale
        
        if return_center:
            return p, centroids
        else:
            return p
    
    def update_mean(self, aligned_pointsets, normalize=True):
        """
        Description: Finds the optimal shape mean given the aligned pointsets.
        
        Inputs:
            aligned_pointsets [DxNxM]: The pointsets aligned using optimize1.m or optimize2.m
            normalize [boolean]: Specifies whether to normalize the mean or not. 
                                If we work in the preshape space, this is required, 
                                but it's not needed in the second approach.
        
        Outputs:
            mu [DxN]: The mean pointset.
        """
        
        # Calculate the mean of the aligned pointsets
        mu = np.mean(aligned_pointsets, axis=2)
        
        # Normalize the mean if required
        if normalize:
            temp = mu ** 2
            scale = np.sqrt(np.sum(temp))
            mu = mu / scale
        
        return mu
    
    def _find_closest_point_pair(self, segments):
        min_distance = float('inf')
        
        for i, segment1 in enumerate(segments):
            for j, segment2 in enumerate(segments):
                if i != j:  # Skip comparing a segment with itself
                    distance_matrix = cdist(segment1, segment2, 'euclidean')
                    min_dist = np.min(distance_matrix)
                    min_index = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)

                    if min_dist < min_distance:
                        min_distance = min_dist
                        closest_pair = np.array([None]*len(segments))
                        closest_pair[i] = segment1[min_index[0]]
                        closest_pair[j] = segment2[min_index[1]]
                        closest_pair_index = np.array([None]*len(segments))
                        closest_pair_index[i] = min_index[0]
                        closest_pair_index[j] = min_index[1]

        return closest_pair, closest_pair_index, min_distance

    def reorganize(self, ):
        with open(self.mean_shape_savep, 'r') as f:
            mean_shape_seg = json.load(f)
        
        with open(self.mean_concatpts_savep, 'r') as f:
            concat_pts_dict = json.load(f)

        # get mean positions of RCA ostium and LAD ostium
        # with open(self.root+'average_shape_results/RCA_points_all.json', 'r') as f:
        #     RCA_points_all = json.load(f)
        
        # with open(self.root+'average_shape_results/LAD_points_all.json', 'r') as f:
        #     LAD_points_all = json.load(f)

        # rca_ostium = np.mean(np.array(RCA_points_all['pointset']), axis=0)[0,:]
        # lad_ostium = np.mean(np.array(LAD_points_all['pointset']), axis=0)[0,:]

        rca_ostium = np.array([ 80.81093459,  80.57357031, -52.15868247]) # from our cohort
        lad_ostium = np.array([ 92.20710637, 105.93229424, -39.00197082]) # from our cohort

        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(121, projection='3d')
        # ax2 = fig.add_subplot(222, projection='3d')
        ax3 = fig.add_subplot(122, projection='3d')
        [self.plot_shape(ax1, np.array(mean_shape_seg[k]).T, self.color_dict[k]) for k in mean_shape_seg.keys()]
        
        # concat_pts = []
        for k1 in concat_pts_dict.keys():
            conca_pt = concat_pts_dict[k1][list(concat_pts_dict[k1].keys())[0]]
            ax1.scatter3D(conca_pt[0], conca_pt[1], conca_pt[2], marker='o', color='r', s=20)
        
        mean_shape_tree = copy.deepcopy(mean_shape_seg)
        # 4. translate RCA and LAD referred to mean positions of ostiums
        r_v = rca_ostium - np.array(mean_shape_seg['RCA'])[0, :]
        new_right_sets = np.array(mean_shape_seg['RCA']) + r_v.reshape(1, -1)
        mean_shape_tree.update({'RCA': new_right_sets.tolist()})

        l_v = lad_ostium - np.array(mean_shape_seg['LAD'])[0, :]
        new_left_sets = np.array(mean_shape_seg['LAD']) + l_v.reshape(1, -1)
        mean_shape_tree.update({'LAD': new_left_sets.tolist()})
        # [self.plot_shape(ax2, np.array(mean_shape_tree[k]).T, self.color_dict[k]) for k in mean_shape_tree.keys()]
        # plt.show()

        moved = []
        new_concat_dict = {}
        for sn in ['LAD', 'LCX', 'RCA']: # sn: main branches
            # find related keys
            k1s = []
            k2s = []
            for k1 in concat_pts_dict.keys():
                if sn in k1:
                    k1s.append(k1)
                    k2s.append(list(concat_pts_dict[k1].keys())[0])
                
                if (sn == 'LAD') and (sn in list(concat_pts_dict[k1].keys())[0]):
                    k1s.append(k1)
                    k2s.append(list(concat_pts_dict[k1].keys())[0])
            
            for j, key in enumerate(k2s): # key: to be moved subbranches
                tgp_index = concat_pts_dict[k1s[j]]['pt_index']
                tgp = mean_shape_tree[sn][int(tgp_index)] # 
                
                ax3.scatter3D(tgp[0], tgp[1], tgp[2], marker='o', color='r', s=20)
                # assert tgp == mean_shape_seg[sn][int(tgp_index)]
                new_concat_dict.update({k1s[j]: {key:tgp, 'pt_index':tgp_index}})
                if key == 'LAD, LCX, RAMUS': 
                    for k in key.split(', ')[1:]:
                        ctl_concate = mean_shape_seg[k]

                        translation_vector = tgp - np.array(ctl_concate)[0, :]
                        added_seg_points = np.array(ctl_concate) + translation_vector.reshape(-1,1).transpose(1,0)
                        mean_shape_tree.update({k:added_seg_points.tolist()})
                        moved.append(k)
                else:
                    ctl_concate = mean_shape_seg[key]
                    translation_vector = tgp - np.array(ctl_concate)[0, :]
                    added_seg_points = np.array(ctl_concate)+translation_vector.reshape(1, -1)
                    mean_shape_tree.update({key:added_seg_points.tolist()})
                    moved.append(key)

        [self.plot_shape(ax3, np.array(mean_shape_tree[k]).T, self.color_dict[k]) for k in mean_shape_tree.keys()]
        plt.show()

        js = json.dumps(mean_shape_tree, indent=4, ensure_ascii=False)
        with open(self.root+'average_shape_results/mean_TREE_shapes.json', 'w') as f:
            f.write(js)
        
        js = json.dumps(new_concat_dict, indent=4, ensure_ascii=False)
        with open(self.root+'average_shape_results/TREE_concat_pts.json', 'w') as f:
            f.write(js)
    
    def show_avg_shapes(self,):
        with open(self.root+'average_shape_results/mean_TREE_shapes.json') as f:
            mean_shape_tree = json.load(f)
        with open(self.mean_shape_savep) as f:
            mean_shape_seg = json.load(f)

        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        [self.plot_shape(ax1, np.array(mean_shape_seg[k]).T, self.color_dict[k]) for k in mean_shape_seg.keys()]
        [self.plot_shape(ax2, np.array(mean_shape_tree[k]).T, self.color_dict[k]) for k in mean_shape_tree.keys()]
        
        # Normalize the color_dict for the colormap
        cmap = mcolors.ListedColormap(self.colors[:len(self.segments)])
        norm = mcolors.BoundaryNorm(boundaries=range(len(self.segments) + 1), ncolors=len(self.segments))
        # Add a color bar for the figure
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Dummy data for the color bar
        cbar = fig.colorbar(sm, ax=[ax1, ax2], shrink=0.6, aspect=20)
        cbar.set_ticks(range(len(self.segments)))
        cbar.set_ticklabels(self.segments)
        cbar.set_label("Segments")
        
        plt.show()
        
    def plot_shape(self, ax, P, clr, label='A', dim=3,**plotargs):
        """Plot the shape with coordinates in the numpy array P 
        as [x1,x2,x3,..,xp,y1,y2,...,yp] or 3*N
        Keyword arguments are passed directly to plot command.
        """
        if P.ndim == 2:
            P = P.flatten()
        N = len(P)
        k = int(N/dim)
        x = P[:k]
        y = P[k:2*k]
        z = P[2*k:]

        ax.scatter3D(x, y, z, marker='o', color=clr, s=8) # , label=label
        ax.plot(x, y, z, color=clr, label=label, lw=3)
        # ax.set_facecolor('black')
    
    def execute(self):
        '1. resampling centerlines'
        self.ctl_resampling()
        
        '2. mean shape of each independent segment'
        self.preshape_mu_seg()

        '3. reorganize mean shapes of all segments according to intersections'
        self.reorganize()
        
        '4. show independent mean segments & average shapes'
        self.show_avg_shapes()

class Plaque_mapping():
    def __init__(self):
        super().__init__()
        self.dataset_size = 16300
        self.dataset_root = './sample_data/'  
        self.result_root = './results_demo/' 

        self.feature_root = Path(self.dataset_root)/'coronary_lesion/'
        self.ctl_path = self.dataset_root + 'centerline_internal/'
        self.lm_ctl_path = self.dataset_root + 'sCPR_ctl_lm/'

        self.AHA_segments = ['RCA', 'R-PLB', 'R-PDA', 'LM', 'LAD', 'LCX', 'RAMUS', 'OM1', 'OM2', 'D1', 'D2', 'L-PLB', 'L-PDA']
        self.AHA18_segments = ['R-PLB', 'R-PDA', 'LM', 'RAMUS', 'OM1', 'OM2', 'D1', 'D2', 'L-PLB', 'L-PDA', \
                               'RCA-Proximal', 'RCA-Middle', 'RCA-Distal', 'LAD-Proximal', 'LAD-Middle', 'LAD-Distal', 'LCX-Proximal', 'LCX-Distal']

        self.key_dict={'stenosis':'stenosis_percent (%)', 
                        'remodeling_index':'remodeling_index (%)', 
                        'eccentricity_index':'eccentricity_index (%)', 
                        'high_risk features':['positive_remodeling', 'low_attenuation_plaque', 'napkin_ring_sign', 'spotty_calcification']}
   
    def position_mapping_all(self, save_path, type_agg='age',pop_select='all', feature_path='./dataset/intgrated_feature_table_16300.csv',
                              tmp_name='mean_TREE_shapes', seg_map_list=['LM', 'LAD', 'LCX', 'RCA'], key_dict=None):
        '''
        save_path: defines location to save the mapped results
        feature_path: path to the plaque characteristics table (csv)
        tmp_name: file name of mean centerline shape, constant. 'mean_TREE_shapes'
        seg_map_list: list of branches to be mapped
        key_list: None: save plaque existence mapping results
                not None: save other characteristics mapping results
            
        other inputs: segment-level results seg_num.json as mapsize

        Output: plaque prevalence distribution (**_plaexist_dict.json) 
                OR aggregated characteristics sets at each point of the mean shape (**_{char_name}_dict.json)
        '''
        if key_dict==None:
            (Path(save_path)/'Plaque_existence/').mkdir(parents=True, exist_ok=True)
        else:
            for char_name in key_dict.keys():
                (Path(save_path)/Path(char_name)).mkdir(parents=True, exist_ok=True)

        dataset_size = self.dataset_size
        cohort_info = pd.read_csv(self.dataset_root+'cohort.csv')

        # --plaque features: have not included healthy subjects
        table = pd.read_csv(feature_path)

        # population seletion
        if pop_select == 'all':
            cohort_info = cohort_info
            filtered_df = table
        elif pop_select == 'female':
            cohort_info = cohort_info[cohort_info['Gender']=='F']
            filtered_df = table[table['Gender']=='F']
        elif pop_select == 'male':
            cohort_info = cohort_info[cohort_info['Gender']=='M']
            filtered_df = table[table['Gender']=='M']
        
        # Group by 'Age Group' and 'Gender'
        age_bins = [0, 44, 54, 64, 74, 150]
        age_labels = ["<45", "45-55", "55-65", "65-75", ">=75"]
        # age_bins = [0] +list(range(44, 80, 5)) + [100]
        # age_labels = ["<45", "45-50", "50-55", "55-60", "60-65", "65-70", "70-75", "75-80", ">=80"]
        
        filtered_df['Age Group'] = pd.cut(filtered_df['Age_imaging'], bins=age_bins, labels=age_labels)

        if type_agg == 'age':
            templates = pd.DataFrame({'Age Group':age_labels, 'Gender':[pop_select]*len(age_labels), 'Subject':[tmp_name]*len(age_labels)})
            grouped = filtered_df.groupby(['Age Group'])

            cohort_info['Age Group'] = pd.cut(cohort_info['Age_imaging'], bins=age_bins, labels=age_labels)
            grouped_ch = cohort_info.groupby(['Age Group'])
            
        
        elif type_agg == 'gender':
            templates = pd.DataFrame({'Age Group':['all', 'all'], 'Gender':['F', 'M'], 'Subject':[tmp_name, tmp_name]})
            grouped = filtered_df.groupby(['Gender'])  # grouped plaque info
            grouped_ch = cohort_info.groupby('Gender')

        elif type_agg == 'all':
            templates = pd.DataFrame({'Age Group':['all'], 'Gender':['all'], 'Subject':[tmp_name]})
            grouped = filtered_df

        
        # --Calculate plaque number in all age group
        # -- centerlines
        tmp_path = self.dataset_root+'average_shape_results/'
        ctl_path = self.dataset_root+'sCPR_ctl_lm/'
        # --intersection 
        bif_path = self.dataset_root+'intersection/'
        # ---
        for i in range(len(templates)):
            tmp_row = templates.iloc[i]
            tmp_subname = tmp_row['Subject']
            
            with open(Path(tmp_path)/(tmp_subname+'.json')) as f:
                tmp_ctl = json.load(f)
            # --load bifurcation file
            with open(Path(tmp_path)/('TREE_concat_pts.json')) as f:
                tmp_bif = json.load(f)

            # ---load map_size.json from segmen-level results 
            map_size_path = Path(save_path).parent.parent/'segment_level/'
            if tmp_row.values[0]=='all' and tmp_row.values[1]=='all':
                with open(map_size_path/'all_seg_num.json', 'r') as f:
                    map_size = json.load(f)
            elif tmp_row.values[0]=='all' and tmp_row.values[1]!='all':
                with open(map_size_path/f'{tmp_row.values[1]}_seg_num.json', 'r') as f:
                    map_size = json.load(f)
            elif tmp_row.values[0]!='all':
                if tmp_row.values[1]=='all':
                    t = 'all'
                else:
                    t = tmp_row.values[1][0].upper()
                with open(map_size_path/f'{t}_age_seg_num.json', 'r') as f:
                    age_ms = json.load(f)
                map_size = age_ms[tmp_row.values[0]]

            # ---split LAD into LAD and LM
            lm_ctl = tmp_ctl['LAD'][0:int(tmp_bif['LM']['pt_index'])]
            lad_ctl = tmp_ctl['LAD'][int(tmp_bif['LM']['pt_index'])::]
            tmp_ctl.update({'LM': lm_ctl})
            tmp_ctl.update({'LAD': lad_ctl})
            
            if type_agg == 'age':
                tmp_age = tmp_row['Age Group']
                tmp_gender = pop_select
                
            elif type_agg =='age_gender':
                tmp_age = tmp_row['Age Group']
                tmp_gender = tmp_row['Gender']
            elif type_agg == 'gender':
                tmp_age = 'all'
                tmp_gender = tmp_row['Gender']
            elif type_agg == 'all':
                tmp_age = 'all'
                tmp_gender = 'all'

            # select segs that needs to be count
            dense_tmp_ctl = {key:[] for key in seg_map_list}
            dense_tmp_ctl_length = {key:[] for key in seg_map_list}


            if key_dict == None:
                dense_count_tree = {key:[] for key in seg_map_list}
            else:
                mapping_tree = {char_name: {key:[] for key in seg_map_list} for char_name in key_dict.keys()}                

            for seg_name in seg_map_list:
                dense_tmp_ctl[seg_name] = tmp_ctl[seg_name]
                dense_tmp_ctl_length[seg_name] = centerline_length2(tmp_ctl[seg_name])
                if key_dict == None:
                    dense_count_tree[seg_name] = [0]*len(tmp_ctl[seg_name])
                else:
                    for char_name in key_dict.keys():
                        mapping_tree[char_name][seg_name] = [[] for _ in range(len(tmp_ctl[seg_name]))]
                print(f'Template {seg_name} with length {dense_tmp_ctl_length[seg_name]}')
            
            # ----------start iterate plaques-------------
            # Iterate through each group
            if type_agg == 'age':
                group_df = grouped.get_group((tmp_age))
                print(f"Processing group with Age: {tmp_age}, Gender: {pop_select}")
            elif type_agg == 'gender':
                group_df = grouped.get_group((tmp_gender))
                print(f"Processing all Age, Gender: {tmp_gender}")
            elif type_agg == 'all': 
                group_df = grouped
                print(f"Processing group with Age: {tmp_age}, Gender: {tmp_gender}")
            
            # next_subj = None
            # counted_subjs = []
            for index, row in tqdm(group_df.iterrows(), total=group_df.shape[0]):

                parsing = row['parsing']
                if parsing not in self.AHA18_segments:
                    continue
                
                if parsing in ['RCA-Proximal', 'RCA-Middle', 'RCA-Distal', 'LAD-Proximal', 'LAD-Middle', 'LAD-Distal', 'LCX-Proximal', 'LCX-Distal']:
                    j_seg_name = parsing.split('-')[0]
                else:
                    j_seg_name = parsing
                

                sub_name_j = Path(row['image_name']).stem
                p_j = [row['world_coordinate_point.x'], row['world_coordinate_point.y'], row['world_coordinate_point.z']]
                range_j = row['range (mm)']
                
                # --load ctl of this segment
                with open(Path(ctl_path)/(sub_name_j+'.json')) as f:
                    j_ctl = json.load(f)

                j_seg_ctl =j_ctl[j_seg_name]
                
                if centerline_length2(j_seg_ctl)==0:
                    continue
                # ----
                # --load bifurcation file
                with open(Path(bif_path)/(sub_name_j+'.json')) as f:
                    j_bif = json.load(f)
                
                # get reference bifurcation point from bif files
                p_b = self.get_ref_bifpoint_(j_seg_name, j_bif, j_ctl)
                
                # --boundary point
                _, point_index, m_d = self.find_closest_point_(j_seg_ctl, [p_j])
                
                # Distance to j_bif of each centerline point
                distances, accumulated_distances = self.get_distance_array(j_seg_ctl, point_index[0])
                    
                # -- Calc distance from bif point to all points on tmp ctl
                tmp_p_b = self.get_ref_bifpoint_(j_seg_name, tmp_bif, tmp_ctl)
                _, tmp_bif_index, m_d = self.find_closest_point_(dense_tmp_ctl[j_seg_name], [tmp_p_b])
                tmp_distances, tmp_accumulated_distances = self.get_distance_array(dense_tmp_ctl[j_seg_name], tmp_bif_index[0])
                # ----------------------------------------------
                indices = np.where(np.abs(distances) <= range_j / 2)[0]
                _, bif_index, m_d = self.find_closest_point_(j_seg_ctl, [p_b])
                dist_on_ctl = np.abs(accumulated_distances[indices] - accumulated_distances[bif_index[0]])
                # ----start mapping----------------
                pr_j = dist_on_ctl/centerline_length2(j_seg_ctl)
                # ----
                rela_dist_on_ctl = dense_tmp_ctl_length[j_seg_name]*pr_j
                m_ind_list =list(range(np.argmin(np.abs(tmp_distances-rela_dist_on_ctl[0])), np.argmin(np.abs(tmp_distances-rela_dist_on_ctl[-1]))+1))
                for x, m_ind in enumerate(m_ind_list):
                    # m_ind = np.unravel_index(np.argmin(np.abs(tmp_distances-value)), tmp_distances.shape)[0]
                    if key_dict == None:
                        dense_count_tree[j_seg_name][m_ind]+=1
                    else:
                        for char_name in key_dict.keys():
                            if char_name=='high_risk features':
                                mapping_tree[char_name][j_seg_name][m_ind].append(list(row[key_dict[char_name]].values))
                            else:
                                mapping_tree[char_name][j_seg_name][m_ind].append(row[key_dict[char_name]])

                # ---------------------------------------
                # if plaque at the start of lad
                if (np.abs(distances[0]-0.29) < range_j / 2) and (j_seg_name=='LAD') and (len(j_ctl['LM'])!=0): # 
                    out_range = range_j/2-np.abs(distances[0])
                    distances_sup, accumulated_distances_sup = self.get_distance_array(j_ctl['LM'], -1)
                    indices_sup = np.where(np.abs(distances_sup) <= out_range)[0]
                    dist_on_ctl_sup = np.abs(accumulated_distances_sup[indices_sup] - accumulated_distances_sup[0])
                    
                    pr_j_sup = dist_on_ctl_sup/centerline_length2(j_ctl['LM'])
                    rela_dist_on_ctl_sup = dense_tmp_ctl_length['LM']*pr_j_sup
                    # rela_dist_on_ctl_sup = self.denser_distance(rela_dist_on_ctl_sup)
                    tmp_distances_sup, _ = self.get_distance_array(dense_tmp_ctl['LM'], 0)

                    m_ind_list =list(range(np.argmin(np.abs(tmp_distances_sup-rela_dist_on_ctl_sup[0])), np.argmin(np.abs(tmp_distances_sup-rela_dist_on_ctl_sup[-1]))+1))
                    for x, m_ind in enumerate(m_ind_list):
                        if key_dict == None:
                            dense_count_tree['LM'][m_ind]+=1
                        else:
                            for char_name in key_dict.keys():
                                if char_name=='high_risk features':
                                    mapping_tree[char_name]['LM'][m_ind].append(list(row[key_dict[char_name]].values))
                                else:
                                    mapping_tree[char_name]['LM'][m_ind].append(row[key_dict[char_name]])
                
                elif (np.abs(distances[-1]+0.29) < range_j / 2) and (j_seg_name=='LM'): # at the end of LM
                    out_range = range_j/2-np.abs(distances[-1])
                    distances_sup, accumulated_distances_sup = self.get_distance_array(j_ctl['LAD'], 0)
                    indices_sup = np.where(np.abs(distances_sup) <= out_range)[0]
                    dist_on_ctl_sup = np.abs(accumulated_distances_sup[indices_sup] - accumulated_distances_sup[0])
                    
                    pr_j_sup = dist_on_ctl_sup/centerline_length2(j_ctl['LAD'])
                    rela_dist_on_ctl_sup = dense_tmp_ctl_length['LAD']*pr_j_sup
                    # rela_dist_on_ctl_sup = self.denser_distance(rela_dist_on_ctl_sup)
                    tmp_distances_sup, _ = self.get_distance_array(dense_tmp_ctl['LAD'], 0)

                    m_ind_list =list(range(np.argmin(np.abs(tmp_distances_sup-rela_dist_on_ctl_sup[0])), np.argmin(np.abs(tmp_distances_sup-rela_dist_on_ctl_sup[-1]))+1))
                    for x, m_ind in enumerate(m_ind_list):
                        if key_dict == None:
                            dense_count_tree['LAD'][m_ind]+=1
                        else:
                            for char_name in key_dict.keys():
                                if char_name=='high_risk features':
                                    mapping_tree[char_name]['LAD'][m_ind].append(list(row[key_dict[char_name]].values))
                                else:
                                    mapping_tree[char_name]['LAD'][m_ind].append(row[key_dict[char_name]])
            

            js = json.dumps(dense_tmp_ctl, indent=4, ensure_ascii=False)
            with open(Path(save_path)/(tmp_age.replace('<', '').replace('>=', '')+'_'+tmp_gender+'_'+pop_select+'_tmpctl.json'), 'w') as f:
                f.write(js)
            js2 = json.dumps(map_size, indent=4, ensure_ascii=False, default=convert)
            with open(Path(save_path)/(tmp_age.replace('<', '').replace('>=', '')+'_'+tmp_gender+'_'+pop_select+'_mapsize.json'), 'w') as f:
                f.write(js2)

            if key_dict==None:
                # convert counts to frequency
                for ky in list(dense_count_tree.keys()):
                    if ky in ['LAD', 'LCX', 'RCA']:
                        ky2 = ky+'-Proximal'
                    dense_count_tree[ky] = (np.array(dense_count_tree[ky])/np.array(map_size[ky2])*100).tolist()
                js = json.dumps(dense_count_tree, indent=4, ensure_ascii=False)
                with open((Path(save_path)/'Plaque_existence/')/(tmp_age.replace('<', '').replace('>=', '')+'_'+tmp_gender+'_'+pop_select+'_plaexist_dict_samples.json'), 'w') as f:
                    f.write(js)
            else:
                for char_name in key_dict.keys():
                    js = json.dumps(mapping_tree[char_name], indent=4, ensure_ascii=False)
                    with open(Path(save_path)/Path(char_name)/Path(tmp_age.replace('<', '').replace('>=', '')+'_'+tmp_gender+'_'+pop_select+f'_{char_name}_dict_samples.json'), 'w') as f:
                        f.write(js)
    
    def find_closest_point_(self, centerline, point_list):
        min_distance = float('inf')
        
        closest_pair = np.array([None]*len(point_list))
        closest_pair_index = np.array([None]*len(point_list))
        for i, point in enumerate(point_list):
            distance_matrix = cdist(centerline, [point], 'euclidean')
            min_distance = np.min(distance_matrix)
            min_index = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
            closest_pair[i] = centerline[min_index[0]]
            # closest_pair[j] = point_list[min_index[1]]
            closest_pair_index[i] = min_index[0]

        return closest_pair, closest_pair_index, min_distance    
    
    def get_distance_array(self, seg_ctl, ref_point_ind):
        diffs_sup = np.diff(seg_ctl, axis=0)
        segment_distances_sup = np.linalg.norm(diffs_sup, axis=1) # distance between adjacent centerline points
        accumulated_distances_sup = np.insert(np.cumsum(segment_distances_sup), 0, 0)
        # point indexes within plaque range
        distances_sup = accumulated_distances_sup - accumulated_distances_sup[ref_point_ind]

        return distances_sup, accumulated_distances_sup
        
    def find_closest_point_pair(self, segments):
        min_distance = float('inf')
        
        for i, segment1 in enumerate(segments):
            for j, segment2 in enumerate(segments):
                if i != j:  # Skip comparing a segment with itself
                    distance_matrix = cdist(segment1, segment2, 'euclidean')
                    min_dist = np.min(distance_matrix)
                    min_index = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)

                    if min_dist < min_distance:
                        min_distance = min_dist
                        closest_pair = np.array([None]*len(segments))
                        closest_pair[i] = segment1[min_index[0]]
                        closest_pair[j] = segment2[min_index[1]]
                        closest_pair_index = np.array([None]*len(segments))
                        closest_pair_index[i] = min_index[0]
                        closest_pair_index[j] = min_index[1]

        return closest_pair, closest_pair_index, min_distance
        
    def get_ref_bifpoint_(self, j_seg_name, j_bif_dict, j_ctl):
        if j_seg_name in ['LAD', 'LCX']:
            if 'RAMUS' in list(j_ctl.keys()):
                p_b = j_bif_dict['LM']['LAD, LCX, RAMUS']
            else:
                p_b = j_bif_dict['LM']['LAD, LCX']
        elif j_seg_name in ['D1']:
            p_b = j_bif_dict['LAD-1']['D1']
        elif j_seg_name in ['D2']:
            p_b = j_bif_dict['LAD-2']['D2']
        elif j_seg_name in ['OM1']:
            p_b = j_bif_dict['LCX-1']['OM1']
        elif j_seg_name in ['OM2']:
            p_b = j_bif_dict['LCX-2']['OM2']
        elif j_seg_name in ['L-PLB']:
            p_b = j_bif_dict['LCX-3']['L-PLB']
        elif j_seg_name in ['L-PDA']:
            p_b = j_bif_dict['LCX-4']['L-PDA']
        elif j_seg_name in ['R-PLB']:
            p_b = j_bif_dict['RCA-1']['R-PLB']
        elif j_seg_name in ['R-PDA']:
            p_b = j_bif_dict['RCA-2']['R-PDA']
        else:
            p_b = j_ctl[j_seg_name][0] # LM and RCA
        return p_b
 
    def execute(self):
        feature_path=self.dataset_root+'intgrated_feature_table_samples.csv'
        save_path = self.result_root+'point_level/position_analysis/'
        
        '''
        define mapping groups here. the first column defines grouping type, the second column defines selected population
        grouping = [['all', 'all'], 
                    ['gender', 'all'], 
                    ['age', 'all'], 
                    ['age', 'female'], 
                    ['age', 'male']]
        - Five subjects are given as samples, so only the first grouping=['all', 'all'] can be properly executed.
        - Running other grouping type, more samples and their age and sex information need to be added 
          to the 'feature_path' file.
        '''
        
        # for group in grouping:
        group = ['all', 'all']
        self.position_mapping_all(save_path = save_path, feature_path=feature_path, 
                                       type_agg=group[0], pop_select=group[1], seg_map_list=self.AHA_segments)
            
        # - other characteristics mapping: set key_dict=self.key_dict
        # for group in grouping:
        self.position_mapping_all(save_path = save_path, feature_path=feature_path, key_dict=self.key_dict, 
                                       type_agg=group[0], pop_select=group[1], seg_map_list=self.AHA_segments)

class Visualization():
    def __init__(self, ):
        super().__init__()
        self.dataset_size = 16300
        self.dataset_root = './sample_data/'  
        self.result_root = './results_demo/'

        self.fig_save = './figs/'
        self.key_dict={'stenosis':'stenosis_percent (%)', 
                'remodeling_index':'remodeling_index (%)', 
                'eccentricity_index':'eccentricity_index (%)', 
                'high_risk features':['positive_remodeling', 'low_attenuation_plaque', 'napkin_ring_sign', 'spotty_calcification']}


        '---Create a custom colormap--------'
        colors1 = plt.cm.jet(np.linspace(0, 1, 1000))
        colors2 = plt.cm.gnuplot(np.linspace(0, 1, 1000))    
        color4 = np.hstack((np.linspace(0.35, 0, 50).reshape(50,1),
                    np.linspace(0, 0.00196, 50).reshape(50,1),
                    np.linspace(0.65, 0.997, 50).reshape(50,1),
                    np.ones((50,1))))
        color5 = np.hstack((np.zeros((50,1)),
                    np.linspace(0.00196, 0.5039, 50).reshape(50,1),
                    np.linspace(0.997, 1, 50).reshape(50,1),
                    np.ones((50,1))))
        random.seed(2024)
        arr = np.vstack((colors2[0:125], color4, color5)) 
        print(arr.shape[0])
        # step_size = arr.shape[0] // 100
        ind = np.sort(np.random.choice([i for i in range(arr.shape[0])], 58, replace=False))
        sampled_rows = arr[ind,:]
        print(sampled_rows.shape)
        colors = np.vstack((sampled_rows, colors1[250:]))
        self.selfcmap = LinearSegmentedColormap.from_list('custom_jet', colors, N=colors.shape[0])
        '-------------------------------------------------'

        '--fonts---'
        self.font0 = {'family': 'sans-serif', 
        'weight': 'normal',
        'size': 10,
        }
        self.font1_5 = {'family': 'sans-serif', 
        'weight': 'normal',
        'size': 12,
        }
        self.font1 = {'family': 'sans-serif', 
        'weight': 'normal',
        'size': 14,
        }
        self.font2 = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 16,
        }
        self.font3 = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 18,
        }
        self.font4 = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 20,
        }

    def prevalence_ctl_dist(self, count_root, type_agg='age', pop_select='all', vmax=10, cmap='inferno'):
        '''
        count_root: str, path to the root folder of the count results
        type_agg: str, type of grouping. 
        pop_select: str, selected population.
            groups for these two parameters are:
                ['all', 'all'], 
                ['gender', 'all'], 
                ['age', 'all'], 
                ['age', 'female'], 
                ['age', 'male']]
        vmax: int, maximum value of the colorbar.
        cmap: str, colormap for the visualization.
        '''
        age_labels = ["<45", "45-55", "55-65", "65-75", ">=75"]
        # age_labels = ["<45", "45-50", "50-55", "55-60", "60-65", "65-70", "70-75", "75-80", ">=80"]

        genders = ['F', 'M']
        
        vmin=0
        point_size = 40
        if type_agg == 'age':
            for idx, age in enumerate(age_labels):
                # ax = fig.add_subplot(1, 5, idx+1, projection='3d')
                age = age.replace('<', '').replace('>=', '')
                with open((Path(count_root)/'Plaque_existence')/(f'{age}_{pop_select}_{pop_select}_plaexist_dict.json')) as f: # +'posterio_probs/'
                    preva = json.load(f)
                with open(Path(count_root)/(f'{age}_{pop_select}_{pop_select}_tmpctl.json')) as f:
                    tmpctl = json.load(f)
                # Create a figure
                # Loop through each preve and tmpctl
                for keys in [['RCA', 'R-PLB', 'R-PDA'], ['LM', 'LAD', 'LCX', 'RAMUS', 'OM1', 'OM2', 'D1', 'D2', 'L-PLB', 'L-PDA']]:
                    values = []
                    pointcloud = []
                    if len(keys)==3:
                        cor = 'R'
                        elev, azim = 30, -60
                    else:
                        cor = 'L'
                        elev, azim = 47, 34
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    # print(ax.azim, ax.elev)
                    ax.view_init(elev=elev, azim=azim)
                    for key in keys: # preva.keys()
                        points = np.array(tmpctl[key])  # Extracting 3D points
                        freqs = preva[key]              # Extracting corresponding frequencies
                        values+=freqs
                        pointcloud+=tmpctl[key]
                        # Scatter the points
                        sc = ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], c=np.array(freqs), cmap=cmap, s=point_size, vmin=vmin, vmax=vmax)
                        ax.set_title(f"Population: {pop_select}, Age Group: {age}", fontdict=self.font1)
                        # scatters.append(sc)

                    # Customize tick parameters
                    ax.tick_params(axis='x', which='major', labelsize=self.font1['size'], labelcolor='k')
                    ax.tick_params(axis='y', which='major', labelsize=self.font1['size'], labelcolor='k')
                    ax.tick_params(axis='z', which='major', labelsize=self.font1['size'], labelcolor='k')

                    # Add a shared colorbar on the right
                    cbar = fig.colorbar(sc, location='right', fraction=0.1, pad=0.1, aspect=40)
                    cbar.set_label('×0.01 Probability', rotation=270, labelpad=15, fontdict=self.font2)
                    cbar.ax.tick_params(labelsize=self.font1['size'], labelcolor='k')

                    ((Path(count_root)/'Plaque_existence')/('plots_views')).mkdir(parents=True, exist_ok=True)
                    # plt.show()
                    plt.savefig((Path(count_root)/'Plaque_existence')/Path('plots_views')/(f'{age}_{pop_select}_{pop_select}_{elev}_{azim}_{cor}_plot.png'), dpi=300) #, dpi=300
                    plt.close()


        elif type_agg == 'gender':
            age = 'all'
            for idx, gender in enumerate(genders):
                # gender = 'M'
                with open((Path(count_root)/'Plaque_existence')/(f'{age}_{gender}_all_plaexist_dict.json')) as f:
                    preva = json.load(f)
                with open(Path(count_root)/(f'{age}_{gender}_{pop_select}_tmpctl.json')) as f:
                    tmpctl = json.load(f)

                # Create a figure
                # Loop through each preve and tmpctl
                for keys in [['RCA', 'R-PLB', 'R-PDA'], ['LM', 'LAD', 'LCX', 'RAMUS', 'OM1', 'OM2', 'D1', 'D2', 'L-PLB', 'L-PDA']]:
                    values = []
                    pointcloud = []
                    if len(keys)==3:
                        cor = 'R'
                        elev, azim = 30, -60
                    else:
                        cor = 'L'
                        elev, azim = 47, 34
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    ax.view_init(elev=elev, azim=azim)
                    values = []
                    pointcloud = []
                    
                    for key in keys:
                        points = np.array(tmpctl[key])  # Extracting 3D points
                        freqs = preva[key]              # Extracting corresponding frequencies
                        values+=freqs
                        pointcloud+=tmpctl[key]
                        # Scatter the points
                        sc = ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], c=np.array(freqs), cmap=cmap, s=point_size, vmin=vmin, vmax=vmax)
                        # ax.set_title(f"Population: {pop_select}, Gender Group: {gender}", fontdict=font1)
                        
                    # Customize tick parameters
                    ax.tick_params(axis='x', which='major', labelsize=self.font1['size'], labelcolor='k')
                    ax.tick_params(axis='y', which='major', labelsize=self.font1['size'], labelcolor='k')
                    ax.tick_params(axis='z', which='major', labelsize=self.font1['size'], labelcolor='k')

                    # Add a shared colorbar on the right
                    cbar = fig.colorbar(sc, location='right', fraction=0.1, pad=0.1, aspect=40)
                    cbar.set_label('×0.01 Probability', rotation=270, labelpad=15, fontdict=self.font2)
                    cbar.ax.tick_params(labelsize=self.font1['size'], labelcolor='k')
                    
                    ((Path(count_root)/'Plaque_existence')/('plots_views')).mkdir(parents=True, exist_ok=True)
                    plt.savefig((Path(count_root)/'Plaque_existence')/Path('plots_views')/(f'{age}_{gender}_{pop_select}_{elev}_{azim}_{cor}_plot.png'), dpi=300) #, dpi=300
                    plt.close()
            # plt.show()
        
        elif type_agg == 'all':
            with open((Path(count_root)/'Plaque_existence')/'all_all_all_plaexist_dict.json') as f:
                preva = json.load(f)
            with open(Path(count_root)/('all_all_all_tmpctl.json')) as f:
                tmpctl = json.load(f)

            # ----------------
            # fig = plt.figure(figsize=(10, 8))
            # ax = fig.add_subplot(111, projection='3d')
            # fig.patch.set_facecolor('white')
            # elev, azim = 30, -60
            # ax.view_init(elev=elev, azim=azim)

            for keys in [['RCA', 'R-PLB', 'R-PDA'], ['LM', 'LAD', 'LCX', 'RAMUS', 'OM1', 'OM2', 'D1', 'D2', 'L-PLB', 'L-PDA']]:
                values = []
                pointcloud = []
                if len(keys)==3:
                    cor = 'R'
                    elev, azim = 30, -60
                else:
                    cor = 'L'
                    elev, azim = 47, 34

                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.view_init(elev=elev, azim=azim)
                values = []
                pointcloud = []
                for key in keys:
                    points = np.array(tmpctl[key])  # Extracting 3D points
                    freqs = preva[key]              # Extracting corresponding frequencies
                    values+=freqs
                    pointcloud+=tmpctl[key]
                    # Scatter the points
                    sc = ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], c=freqs, cmap=cmap, s=point_size, vmin=vmin, vmax=vmax)
                    # ax.set_title(f"Population: {pop_select}, Gender Group: {type_agg}", fontdict=font2)
                    # scatters.append(sc)
                # plt.xticks(fontfamily=font2['family'], fontweight=font2['weight'], fontsize=font2['size'])
                # plt.yticks(fontfamily=font2['family'], fontweight=font2['weight'], fontsize=font2['size'])
                # Set labels with font properties
                # ax.set_xlabel('X Label', fontdict=font2)
                # ax.set_ylabel('Y Label', fontdict=font2)
                # ax.set_zlabel('Z Label', fontdict=font2)

                # Customize tick parameters
                ax.tick_params(axis='x', which='major', labelsize=self.font1['size'], labelcolor='k')
                ax.tick_params(axis='y', which='major', labelsize=self.font1['size'], labelcolor='k')
                ax.tick_params(axis='z', which='major', labelsize=self.font1['size'], labelcolor='k')

                # Add a shared colorbar on the right
                cbar = fig.colorbar(sc, location='right', fraction=0.1, pad=0.1, aspect=40)
                cbar.set_label('×0.01 Probability', rotation=270, labelpad=15, fontdict=self.font2)
                cbar.ax.tick_params(labelsize=self.font1['size'], labelcolor='k')

                # ax.grid(False)

                ((Path(count_root))/('Plaque_existence/plots_views')).mkdir(parents=True, exist_ok=True)
                plt.savefig((Path(count_root))/('Plaque_existence/plots_views')/(f'all_all_all_{elev}_{azim}_{cor}_plot.png'), dpi=300) #, dpi=300
                # plt.savefig((Path(save_root))/('Plaque_existence/plots_views')/(f'all_all_all_{elev}_{azim}_two_plot.pdf')) # , dpi=300
                plt.close()

    def char_ctl_dist(self, count_root, type_agg='age', pop_select='all', char_name='high_risk features', cmap='inferno'):
        '''
        count_root: str, path to the root folder of the count results
        type_agg: str, type of grouping. 
        pop_select: str, selected population.
            groups for these two parameters are:
                ['all', 'all'], 
                ['gender', 'all'], 
                ['age', 'all'], 
                ['age', 'female'], 
                ['age', 'male']]
        char_name: str, name of the characteristic to be visualized. 'high_risk features', 'stenosis'.
        vmax: int, maximum value of the colorbar.
        cmap: str, colormap for the visualization.
        '''
        age_labels = ["<45", "45-55", "55-65", "65-75", ">=75"]
        # age_labels = ["<45", "45-50", "50-55", "55-60", "60-65", "65-70", "70-75", "75-80", ">=80"]

        genders = ['F', 'M']
        vmin=0
        point_size = 40
        vmax_hr=60
        vmax_ss=60

        if type_agg == 'age':
            for idx, age in enumerate(age_labels):
                age = age.replace('<', '').replace('>=', '')

                with open(Path(count_root)/(f'{age}_{pop_select}_{pop_select}_tmpctl.json')) as f:
                    tmpctl = json.load(f)

                # # ---post processing mapped info---
                # with open(Path(count_root)/Path(char_name)/(f'{age}_{pop_select}_{pop_select}_{char_name}_dict.json')) as f:
                #     map_info = json.load(f)
                # with open(Path(count_root)/(f'{age}_{pop_select}_{pop_select}_mapsize.json')) as f:
                #     map_size = json.load(f)
                # to_draw = self.get_draw_dict(char_name, map_info, map_size)
                # js = json.dumps(to_draw, indent=4, ensure_ascii=False)
                # with open(Path(count_root)/Path(char_name)/Path(age+'_'+pop_select+'_'+pop_select+f'_{char_name}_probs.json'), 'w') as f:
                #     f.write(js)
                # # -----------Replace------------
                with open(Path(count_root)/Path(char_name)/Path(age+'_'+pop_select+'_'+pop_select+f'_{char_name}_probs.json'), 'r') as f:
                    to_draw = json.load(f)
                
                # Create a figure
                # Loop through each preve and tmpctl
                for key in to_draw.keys():
                    if char_name=='high_risk features':
                        if (key not in self.key_dict[char_name]+['HR_pla_preva']):
                            vmax=100
                        else:
                            vmax=vmax_hr
                    if char_name=='stenosis':
                        if '_exist' in key:
                            vmax=100
                        else:
                            vmax=vmax_ss
                    for keys in [['RCA', 'R-PLB', 'R-PDA'], ['LM', 'LAD', 'LCX', 'RAMUS', 'OM1', 'OM2', 'D1', 'D2', 'L-PLB', 'L-PDA']]:
                        if len(keys)==3:
                            cor = 'R'
                            elev, azim = 30, -60
                        else:
                            cor = 'L'
                            elev, azim = 47, 34

                        fig = plt.figure(figsize=(10, 8))
                        ax = fig.add_subplot(111, projection='3d')
                        # print(ax.azim, ax.elev)
                        ax.view_init(elev=elev, azim=azim)
                        for seg_name in keys:
                            points = np.array(tmpctl[seg_name])  # Extracting 3D points
                            freqs = to_draw[key][seg_name]              # Extracting corresponding frequencies
                            # values+=freqs
                            # pointcloud+=tmpctl[key]
                            # Scatter the points
                            sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=np.array(freqs), cmap=cmap, s=point_size, vmin=vmin, vmax=vmax)
                            # ax.set_title(f"{key}, Population: {pop_select}, Age Group: {age}", fontdict=font1)
                        # scatters.append(sc)
                        # Customize tick parameters
                        ax.tick_params(axis='x', which='major', labelsize=self.font1['size'], labelcolor='k')
                        ax.tick_params(axis='y', which='major', labelsize=self.font1['size'], labelcolor='k')
                        ax.tick_params(axis='z', which='major', labelsize=self.font1['size'], labelcolor='k')

                        # Add a shared colorbar on the right
                        cbar = fig.colorbar(sc, location='right', fraction=0.1, pad=0.1, aspect=40)
                        if char_name == 'stenosis' and key == 'mean ste_ratio':
                            cbar.set_label('×0.01 Mean Stenosis Ratio', rotation=270, labelpad=15, fontdict=self.font2)
                        else:
                            cbar.set_label('×0.01 Probability', rotation=270, labelpad=15, fontdict=self.font2)
                        
                        (Path(count_root)/Path(char_name)/Path('plots_views')/Path(key)).mkdir(parents=True, exist_ok=True)
                        # plt.show()
                        plt.savefig(Path(count_root)/Path(char_name)/Path('plots_views')/Path(key)/(f'{age}_{pop_select}_{pop_select}_{elev}_{azim}_{cor}_plot.png'), dpi=300) # , dpi=300
                        plt.close()
                # if save_ply:
                #     self.save_nii_(pointcloud, values, path_to=Path(count_root+'posterio_probs/')/Path('plots')/(f'{age}_{pop_select}_{pop_select}.nii.gz'), cmap=cmap)
                    # self.save_ply_(pointcloud, values, path_to=Path(count_root+'posterio_probs/')/Path('plots')/(f'{age}_{pop_select}_{pop_select}.ply'), cmap=cmap)
            # plt.show()

        elif type_agg == 'gender':
            age = 'all'
            for idx, gender in enumerate(genders):
                
                with open(Path(count_root)/(f'{age}_{gender}_{pop_select}_tmpctl.json')) as f:
                    tmpctl = json.load(f)
                
                # # ---post processing mapped info---
                # with open(Path(count_root)/Path(char_name)/(f'{age}_{gender}_{age}_{char_name}_dict.json')) as f:
                #     map_info = json.load(f)
                # with open(Path(count_root)/(f'{age}_{gender}_{age}_mapsize.json')) as f:
                #     map_size = json.load(f)
                # to_draw = self.get_draw_dict(char_name, map_info, map_size)
                # js = json.dumps(to_draw, indent=4, ensure_ascii=False)
                # with open(Path(count_root)/Path(char_name)/Path(age.replace('<', '').replace('>=', '')+'_'+gender+'_'+pop_select+f'_{char_name}_probs.json'), 'w') as f:
                #     f.write(js)
                # # -----------Replace------------
                with open(Path(count_root)/Path(char_name)/Path(age.replace('<', '').replace('>=', '')+'_'+gender+'_'+pop_select+f'_{char_name}_probs.json'), 'r') as f:
                    to_draw = json.load(f)
                
                for key in to_draw.keys():
                    if char_name=='high_risk features':
                        if (key not in self.key_dict[char_name]+['HR_pla_preva']):
                            vmax=100
                        else:
                            vmax=vmax_hr
                    if char_name=='stenosis':
                        if '_exist' in key:
                            vmax=100
                        else:
                            vmax=vmax_ss
                # fig = plt.figure(figsize=(10, 8))
                # ax = fig.add_subplot(111, projection='3d')
                # ax.view_init(elev=elev, azim=azim)
                    
                    for keys in [['RCA', 'R-PLB', 'R-PDA'], ['LM', 'LAD', 'LCX', 'RAMUS', 'OM1', 'OM2', 'D1', 'D2', 'L-PLB', 'L-PDA']]:
                        if len(keys)==3:
                            cor = 'R'
                            elev, azim = 30, -60
                        else:
                            cor = 'L'
                            elev, azim = 47, 34
                        fig = plt.figure(figsize=(10, 8))
                        ax = fig.add_subplot(111, projection='3d')
                        ax.view_init(elev=elev, azim=azim)
                        for seg_name in keys:
                            points = np.array(tmpctl[seg_name])  # Extracting 3D points
                            freqs = to_draw[key][seg_name]              # Extracting corresponding frequencies
                            # values+=freqs
                            # pointcloud+=tmpctl[key]
                            # Scatter the points
                            sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=np.array(freqs), cmap=cmap, s=point_size, vmin=vmin, vmax=vmax)
                            # ax.set_title(f"{key}, Population: {pop_select}, Gender Group: {gender}", fontdict=font1)
                        # scatters.append(sc)
                        # Customize tick parameters
                        ax.tick_params(axis='x', which='major', labelsize=self.font1['size'], labelcolor='k')
                        ax.tick_params(axis='y', which='major', labelsize=self.font1['size'], labelcolor='k')
                        ax.tick_params(axis='z', which='major', labelsize=self.font1['size'], labelcolor='k')


                        # Add a shared colorbar on the right
                        cbar = fig.colorbar(sc, location='right', fraction=0.1, pad=0.1, aspect=40)
                        if char_name == 'stenosis' and key == 'mean ste_ratio':
                            cbar.set_label('×0.01 Mean Stenosis Ratio', rotation=270, labelpad=15, fontdict=self.font2)
                        else:
                            cbar.set_label('×0.01 Probability', rotation=270, labelpad=15, fontdict=self.font2)
                        
                        (Path(count_root)/Path(char_name)/Path('plots_views')/Path(key)).mkdir(parents=True, exist_ok=True)
                        # plt.show()
                        plt.savefig(Path(count_root)/Path(char_name)/Path('plots_views')/Path(key)/(f'{age}_{gender}_{pop_select}_{elev}_{azim}_{cor}_plot.png'), dpi=300)
                        plt.close()
        
        elif type_agg == 'all':
            # fig = plt.figure(figsize=(10, 8))
            # ax = fig.add_subplot(111, projection='3d')
            # ax.view_init(elev=elev, azim=azim)
            
            with open(Path(count_root)/('all_all_all_tmpctl.json')) as f:
                tmpctl = json.load(f)
            
            # # ---post processing mapped info---
            # with open(Path(count_root)/Path(char_name)/(f'all_all_all_{char_name}_dict.json')) as f:
            #     map_info = json.load(f)
            # with open(Path(count_root)/('all_all_all_mapsize.json')) as f:
            #     map_size = json.load(f)
            # to_draw = self.get_draw_dict(char_name, map_info, map_size)
            # js = json.dumps(to_draw, indent=4, ensure_ascii=False)
            # with open(Path(count_root)/Path(char_name)/Path('all_all_all'+f'_{char_name}_probs.json'), 'w') as f:
            #     f.write(js)
            # # -----------Replace------------
            with open(Path(count_root)/Path(char_name)/Path('all_all_all'+f'_{char_name}_probs.json'), 'r') as f:
                to_draw = json.load(f)

            for key in to_draw.keys():
                if char_name=='high_risk features':
                    if (key not in self.key_dict[char_name]+['HR_pla_preva']):
                        vmax=100
                    else:
                        vmax=vmax_hr
                if char_name=='stenosis':
                    if '_exist' in key:
                        vmax=100
                    else:
                        vmax=vmax_ss

                for keys in [['RCA', 'R-PLB', 'R-PDA'], ['LM', 'LAD', 'LCX', 'RAMUS', 'OM1', 'OM2', 'D1', 'D2', 'L-PLB', 'L-PDA']]:
                    if len(keys)==3:
                        cor = 'R'
                        elev, azim = 30, -60
                    else:
                        cor = 'L'
                        elev, azim = 47, 34
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    # print(ax.azim, ax.elev)
                    ax.view_init(elev=elev, azim=azim)
                    for seg_name in keys:
                        points = np.array(tmpctl[seg_name])  # Extracting 3D points
                        freqs = to_draw[key][seg_name]              # Extracting corresponding frequencies
                        # values+=freqs
                        # pointcloud+=tmpctl[key]
                        # Scatter the points
                        sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=np.array(freqs), cmap=cmap, s=point_size, vmin=vmin, vmax=vmax)
                        # ax.set_title(f"{key}, Population: {pop_select}, Gender Group: {type_agg}", fontdict=font3)
                    # scatters.append(sc)
                    # Set labels with font properties
                    # ax.set_xlabel('', fontdict=font2)
                    # ax.set_ylabel('', fontdict=font2)
                    # ax.set_zlabel('', fontdict=font2)

                    # Customize tick parameters
                    ax.tick_params(axis='x', which='major', labelsize=self.font1['size'], labelcolor='k')
                    ax.tick_params(axis='y', which='major', labelsize=self.font1['size'], labelcolor='k')
                    ax.tick_params(axis='z', which='major', labelsize=self.font1['size'], labelcolor='k')

                    # Add a shared colorbar on the right
                    cbar = fig.colorbar(sc, location='right', fraction=0.1, pad=0.1, aspect=40)
                    cbar.ax.tick_params(labelsize=self.font1['size'], labelcolor='k')
                    if char_name == 'stenosis' and key == 'mean ste_ratio':
                        cbar.set_label('×0.01 Mean Stenosis Ratio', rotation=270, labelpad=15, fontdict=self.font2)
                    else:
                        cbar.set_label('×0.01 Probability', rotation=270, labelpad=15, fontdict=self.font2)
                    
                    (Path(count_root)/Path(char_name)/Path('plots_views')/Path(key)).mkdir(parents=True, exist_ok=True)
                    # plt.show()
                    plt.savefig(Path(count_root)/Path(char_name)/Path('plots_views')/Path(key)/(f'all_all_all_{elev}_{azim}_{cor}_plot.png'), dpi=300) #, dpi=300
                    plt.close()
    
    def get_draw_dict(self, char_name, map_info, map_size):
        to_draw = {}
        valid_pla_TH = 20
        valid_HRpla_TH = 10
        if char_name=='high_risk features':
            # initialize to_draw
            for k, name in enumerate(self.key_dict[char_name]):
                prevalence = {key:[] for key in map_info.keys()}
                for seg_name in map_info.keys():
                    prevalence[seg_name] = [0]*len(map_info[seg_name])
                prevalence2 = copy.deepcopy(prevalence)
                prevalence3= copy.deepcopy(prevalence)
                to_draw.update({name: prevalence})
                to_draw.update({name+'_pla_exist': prevalence2})
                to_draw.update({name+'_HRpla_exist': prevalence3})

            HR_pla_preva = {key:[] for key in map_info.keys()}
            HR_pla_preva_pla_exist = {key:[] for key in map_info.keys()}
            for seg_name in map_info.keys():
                HR_pla_preva[seg_name] = [0]*len(map_info[seg_name])
                HR_pla_preva_pla_exist[seg_name] = [0]*len(map_info[seg_name])
            to_draw.update({'HR_pla_preva': HR_pla_preva})
            to_draw.update({'HR_pla_preva_pla_exist': HR_pla_preva_pla_exist})

            for seg_name in map_info.keys():
                if seg_name in ['RCA', 'LAD', 'LCX']:
                    seg_name2 = seg_name+'-Proximal'
                for ind, mapped_set in enumerate(map_info[seg_name]):
                    if len(mapped_set)==0:
                        continue
                    num_pla = len(mapped_set)
                    
                    # independent prevalence
                    add_count = np.sum(np.array(mapped_set), axis=0)
                    hr_num_list = np.sum(np.array(mapped_set), axis=1)

                    num_HRpla = np.argwhere(hr_num_list>=2).size
                    if num_HRpla != 0:
                        HRpla_set = np.array(mapped_set)[np.where(hr_num_list>=2)]
                    else:
                        HRpla_set = np.array([[0, 0, 0, 0]])

                    # high-risk plaque prevalence
                    to_draw['HR_pla_preva'][seg_name][ind] = np.argwhere(hr_num_list>=2).size/map_size[seg_name2]*100
                    if num_pla>=valid_pla_TH:
                        to_draw['HR_pla_preva_pla_exist'][seg_name][ind] = np.argwhere(hr_num_list>=2).size/num_pla*100
                    else:
                        to_draw['HR_pla_preva_pla_exist'][seg_name][ind] = 0
                    
                    for k, name in enumerate(self.key_dict[char_name]):
                        to_draw[name][seg_name][ind] = add_count[k]/map_size[seg_name2]*100
                        if num_pla>=valid_pla_TH:
                            to_draw[name+'_pla_exist'][seg_name][ind] = add_count[k]/num_pla*100
                        else:
                            to_draw[name+'_pla_exist'][seg_name][ind] = 0

                        if num_HRpla >= valid_HRpla_TH:
                            count_on_HRpla = np.sum(HRpla_set, axis=0)
                            to_draw[name+'_HRpla_exist'][seg_name][ind] = count_on_HRpla[k]/num_HRpla*100
                        else:
                            to_draw[name+'_HRpla_exist'][seg_name][ind] = 0
        # in building
        elif char_name=='stenosis':
            prevalence = {key:[] for key in map_info.keys()}
            for seg_name in map_info.keys():
                prevalence[seg_name] = [0]*len(map_info[seg_name])
            
            mean_ = {key:[] for key in map_info.keys()}
            for seg_name in map_info.keys():
                mean_[seg_name] = [0]*len(map_info[seg_name])
           
            prev_50_plaque = {key:[] for key in map_info.keys()}
            for seg_name in map_info.keys():
                prev_50_plaque[seg_name] = [0]*len(map_info[seg_name])
            
            for seg_name in map_info.keys():
                if seg_name in ['RCA', 'LAD', 'LCX']:
                    seg_name2 = seg_name+'-Proximal'
                for ind, mapped_set in enumerate(map_info[seg_name]):
                    if len(mapped_set)==0:
                        continue
                    # num_ste = len(mapped_set)
                    num_sste = np.argwhere(np.array(mapped_set)>=0.5).size
                    avg_ste = np.sum(mapped_set)/map_size[seg_name2]

                    prevalence[seg_name][ind] = num_sste/map_size[seg_name2]*100
                    mean_[seg_name][ind] = avg_ste*100
                    if len(mapped_set)<valid_pla_TH:
                        prev_50_plaque[seg_name][ind] = 0
                    else:
                        prev_50_plaque[seg_name][ind] = num_sste/len(mapped_set)*100

            to_draw.update({'50% prevalence': prevalence})
            to_draw.update({'mean ste_ratio': mean_})
            to_draw.update({'50% prevalence_pla_exist': prev_50_plaque})
        # --------------------------------

        return to_draw


    def execute(self):
        '''
        Visualize the results of Plaque_mapping().execute()
        Notes: 
        '''
        grouping = [['all', 'all'], 
                    ['gender', 'all'], 
                    ['age', 'all'], 
                    ['age', 'female'], 
                    ['age', 'male']]
        
        # --- for plaque existence visualization----
        for group in grouping:
            self.prevalence_ctl_dist(count_root = self.result_root+'point_level/position_analysis/', 
                                type_agg=group[0], pop_select=group[1], 
                                vmax=70, cmap=self.selfcmap)
        
        # ----for stenosis ratio and high-risk features visualization ---- 
        for char_name in ['stenosis', 'high_risk features']:
            for group in grouping:
                self.char_ctl_dist(count_root = self.result_root+'point_level/position_analysis/', 
                                    type_agg=group[0], pop_select=group[1], 
                                    char_name=char_name,
                                    cmap=self.selfcmap)


if __name__ == '__main__':
    # atlas calculation
    # Averaging_Ctl().execute()

    # # plaque characteristic mapping
    # Plaque_mapping().execute()

    Visualization().execute()
