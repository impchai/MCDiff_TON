# MCDiff_TON



## 1. Data Description and Format

We select **344 grids** from the Nanchang dataset. For each grid, we provide the corresponding satellite images, POI data, population data, mobile traffic, and mobile user statistics. All data are stored in the `data_prepare_nc` folder.

- **Mobile Traffic**  
  - File: `data_prepare_nc/selected_data.npz`  
  - Key: `['observed_traffic']`  
  - Shape: `(344, 168)`  

- **Mobile User**  
  - File: `data_prepare_nc/selected_data.npz`  
  - Key: `['observed_users']`  
  - Shape: `(344, 168)`  

- **Satellite Images**  
  - Folder: `data_prepare_nc/selected_images/`  

- **POI Data**  
  - File: `data_prepare_nc/selected_data.npz`  
  - Key: `['poi_data']`  
  - Shape: `(344, 14)` (14 represents POI categories)  

- **Population Data**  
  - File: `data_prepare_nc/selected_data.npz`  
  - Key: `['human_data']`  
  - Shape: `(344, 12)` (12 represents the number of subregions in each grid)  

---

## 2. Program Workflow

1. **Pre-training**: Contrastive learning is used to align features among satellite images, POI, population, and the mean of mobile traffic/user data.  
2. **Joint Training**: Contrastive learning module and diffusion model are jointly trained.
   
<img src="./framework.png" >

## Getting Started

## 1. Environment

- Python >= 3.7  
- PyTorch >= 2.0.0  
- CUDA >= 11.7  

Use the following command to install all required Python modules and packages:

```bash
pip install -r requirements.txt
 ```

## 2. Running the Code

Run the following command to start training:  

```bash
python exe_traffic_union.py
 ```

## 3. Get Results
After training, a folder named `TrafficData_time` will be generated under the `save/` directory.
