import os
from PIL import Image
from tqdm import tqdm 

def main():
    print("resizing images to 224x224")
    
    source_dir = os.path.join("data", "raw", "images") #main images
    target_dir = os.path.join("data", "processed", "images_224")
    
    os.makedirs(target_dir, exist_ok=True)
    
    images = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]
    
    for img_name in tqdm(images):
        try:
            #open & resize
            with Image.open(os.path.join(source_dir, img_name)) as img:
                img = img.convert('RGB')
                img_resized = img.resize((224, 224), Image.Resampling.LANCZOS)
                
                #save
                img_resized.save(os.path.join(target_dir, img_name), quality=90)
        except Exception as e:
            print(f"error resizing {img_name}: {e}")

    print(f"resized {len(images)} images to {target_dir}")

if __name__ == "__main__":
    main()