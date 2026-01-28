import os

def clean_duplicates():
    #path
    folder_path = os.path.join("data", "raw", "images")
    
    print(f"scanning {folder_path} for duplicates")
    
    count = 0
    #loop through every file
    for filename in os.listdir(folder_path):
        
        # check specifically for the pattern 
        if " (2)" in filename:
            file_path = os.path.join(folder_path, filename)
            
            try:
                os.remove(file_path)
                count += 1
            except Exception as e:
                print(f"Error deleting {filename}: {e}")
                
    print(f"success! Deleted {count} duplicate files.")

if __name__ == "__main__":
    clean_duplicates()