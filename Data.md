All files are available at this google drive folder: https://drive.google.com/drive/folders/1soX0ivDm7GIIVxs1fX6OXIDWEWrLzM6a?usp=sharing. You can add shared folders as  shortcuts to your google drive, then configure [rclone](https://rclone.org/drive/) to access google drive files in terminal. 


# Dataset Download

## iPhone

Please first download the iPhone dataset processed by shape-of-motion from https://drive.google.com/drive/u/1/folders/1xJaFS_3027crk7u36cue7BseAX80abRe. Note that it is much larger than the original iPhone dataset because the 2D track results take a lot of space. 

We provide the instance masks and SAPIENS keypoints for the iPhone dataset, which can be downloaded from `data/iphone` of our drive folder. After they are downloaded, unzip and copy the files to each scene correspondingly. 

The process is like

```bash
# assuming you are in the project root
mkdir datasets && cd ./datasets
rclone copy -P drive:/iphone . # shape-of-motion iphone dataset
cd iphone
for f in *.zip; do unzip $f; done
rm *.zip

rclone copy -P drive:/release/data/iphone . # ours (instance masks and keypoints)
cd ./iphone

for f in *.zip; do 
unzip $f; 
cp -rf ./${f:%.*} ../${f:%.*} # ${f:%.*} remove the .zip extension
done

cd ..
rm -rf ./iphone
```


Run each part / line of the above script carefully, you may need some adjustments depending on your rclone configuration. After the setup, you will have iphone dataset install at `datasets/iphone` folder from your project root. 


## HyperNerf  & Robot

The hypernerf and our robot scenes can be downloaded from the `data/hypernerf` and `data/robot`, and unzip correspondingly to `datasets/hypernerf` and `datasets/robot` folder. 

Note that the hypernerf files are quite large because the 2D track results and I forget to delete the colmap output. 

The robot scene files are much smaller, but also larger than it should because (1) I save all the depth image (actually is disparity) in 32 bit npy file, (2) I only fit the scenes on first few hundreds of frames, but the zip file includes many more frames. 



# Weights Download  

The weight files are located in the `weights` folder of the drive folder, download it with rclone and put them into `outputs/weights` under the project root.  

I did not upload the hypernerf weights because I lost them, but re-fitting them is very easy. 