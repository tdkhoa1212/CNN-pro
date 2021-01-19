import random
from shutil import copyfile

def label_image(img):
    word_label = img.split('.')[0]
    return word_label


def cat_split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    cat_files = []
    for filename in os.listdir(SOURCE):
        name_files = label_image(filename)  # name_files inslude (dog, cat)
        if name_files == 'cat':
            file = SOURCE + filename
            if os.path.getsize(file) > 0:
                cat_files.append(filename)
            else:
                print(filename + "is zero length, so ignoring")
    
    training_length = int(len(cat_files) * SPLIT_SIZE)
    testing_length = int(len(cat_files) - training_length)
    
    shuffled_set = random.sample(cat_files, len(cat_files))
    
    # set train and test
    training_set = shuffled_set[0: training_length]
    testing_set = shuffled_set[testing_length: ]
    
    for filename in training_set:
        this_file = SOURCE + filename
        destination = TRAINING + filename
        copyfile(this_file, destination)
        
    for filename in testing_set:
        this_file = SOURCE + filename
        destination = TESTING + filename
        copyfile(this_file, destination)

def dog_split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    dog_files = []
    for filename in os.listdir(SOURCE):
        name_files = label_image(filename)  # name_files inslude (dog, cat)
        
        if name_files == 'dog':
            file = SOURCE + filename
            if os.path.getsize(file) > 0:
                dog_files.append(filename)
            else:
                print(filename + "is zero length, so ignoring")
        
    training_length = int(len(dog_files) * SPLIT_SIZE)
    testing_length = int(len(dog_files) - training_length)
    
    shuffled_set = random.sample(dog_files, len(dog_files))
    
    # set train and test
    training_set = shuffled_set[0: training_length]
    testing_set = shuffled_set[testing_length: ]
    for filename in training_set:
        this_file = SOURCE + filename
        destination = TRAINING + filename
        copyfile(this_file, destination)
        
    for filename in testing_set:
        this_file = SOURCE + filename
        destination = TESTING + filename
        copyfile(this_file, destination)
        
  # Data of cats
CAT_SOURCE_DIR = "trains/"
TRAINING_CATS_DIR = "train/cats/"
TESTING_CATS_DIR = "test/cats/"

  # Data of dogs
DOG_SOURCE_DIR = "trains/"
TRAINING_DOGS_DIR = "train/dogs/"
TESTING_DOGS_DIR = "test/dogs/"
 
split_size = .9
cat_split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
dog_split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)
