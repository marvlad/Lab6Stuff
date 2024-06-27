# Run the Dockerfile

Create the image, run and execute, example: 

In the path where the Dockerfile is
```javascript
docker build -t acdc ./
```

Run the docker mounting the data file
```javascript
docker run -dit -v /path/of/your/data/file:/data --name my_acdc acdc
```

Get the terminal and run the script 
```javascript
docker exec -it my_acdc /bin/bash
```

Finally
Get the terminal and run the script 
```javascript
cd /acdc && python3 ACDC_WaveForm.py 
```

The output will show up in the `report` directory
 
