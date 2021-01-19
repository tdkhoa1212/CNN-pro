import os
import glob

def rename(file_path, classes):
    for fold in classes:
        old_path = os.path.join(file_path, fold) + '/'
        print(old_path)
        path = os.path.join(file_path, fold, '*g')
        files = glob.glob(path)

        for index, fl in enumerate(files):
            new_name = old_path + fold + '.' + str(index) + '.jpeg'
            os.rename(fl, new_name)

# class info
classes = ['NORMAL', 'PNEUMONIA']
num_classes = len(classes)

train_path = 'train/'
test_path = 'test/'
val_path = 'val/'
checkpoint_dir = "models/"

# rename(val_path, classes)
# rename(test_path, classes)
# rename(train_path, classes)
