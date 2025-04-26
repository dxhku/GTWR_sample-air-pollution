# Geographically and Temporally Weighted Regression (GTWR) Model
## Project Description
This project implements a traditional GTWR model, using Hong Kong NO<sub>2</sub> air pollutant concentration as a case study.

## File Structure
|- Program  
|--- GTWR.py                # GTWR algorithm implementation  
|--- example_data           # Sample data directory  
|----- raster               # Explanatory variable raster images  
|----- shp   
|------- grid_points.shp    # Prediction grid center points (point shapefile)  

## Workflow
1. Explanatory variable collinearity check  
![image](https://github.com/user-attachments/assets/472de155-a046-4ff6-a892-1c0bb39d612b)

2. The best optimal parameter search  
![image](https://github.com/user-attachments/assets/287e01d2-a753-4282-9f25-ba241bd78a99)

3. Accuracy scatter plot generation for variation and fitting results  
![image](https://github.com/user-attachments/assets/62e0c720-1504-4a74-a9fb-dcdb3b24b6ee)

4. Grid prediction calculation with results stored as point shapefiles  
![image](https://github.com/user-attachments/assets/4c833ec6-2466-4e4d-99bf-c8de68c3226a)

## Reference
The algorithm is based on the following literature:
https://doi.org/10.1080/13658810802672469

## Contact
Email: [dengxun@hku.hk]
