
import torch
import torch.nn.functional as F
from torchvision import transforms
from utils import ResizeVideoTransform, VideoToTensorTransform

PREPROCESSING_TRANSFORMS = transforms.Compose([
    ResizeVideoTransform((224, 224)),
    VideoToTensorTransform()
])
NUM_EPOCHS = 4
NUM_CLASSES = 158
NO_ACTION_TENSOR = F.one_hot(torch.tensor(NUM_CLASSES-1), num_classes=NUM_CLASSES)
BATCH_SIZE = 16
LEARNING_RATE = 0.001
GRADIENT_ACCUMULATION_ITERS = 4
PRINT_BATCH_LOSS_EVERY = 4
VIDEO_MAX_SAMPLES = 1
SAMPLE_CLIP_SIZE = 16

CLASS_MAPPING = {
    # Clothing and Dressing (0)
    'c000': 0, 'c001': 0, 'c002': 0, 'c003': 0, 'c004': 0, 'c005': 0, 'c148': 0, 'c155': 0,
    
    # Door/Window Interaction (1)
    'c006': 1, 'c007': 1, 'c008': 1, 'c097': 1, 'c089': 1, 'c090': 1, 'c091': 1, 'c092': 1, 'c140': 1, 'c141': 1, 'c142': 1, 'c143': 1,
    
    # Table Interaction (2)
    'c009': 2, 'c010': 2, 'c011': 2, 'c012': 2, 'c013': 2, 'c014': 2,
    
    # Phone/Camera Interaction (3)
    'c015': 3, 'c016': 3, 'c017': 3, 'c018': 3, 'c019': 3, 'c087': 3, 'c132': 3,
    
    # Bag Handling (4)
    'c020': 4, 'c021': 4, 'c022': 4, 'c023': 4, 'c024': 4,
    
    # Book/Reading (5)
    'c025': 5, 'c026': 5, 'c027': 5, 'c028': 5, 'c029': 5, 'c030': 5, 'c031': 5, 'c032': 5, 'c145': 5, 'c115': 5, 'c116': 5, 'c117': 5,
    
    # Towel Handling (6)
    'c033': 6, 'c034': 6, 'c035': 6, 'c036': 6, 'c037': 6, 'c038': 6,
    
    # Box Interaction (7)
    'c039': 7, 'c040': 7, 'c041': 7, 'c042': 7, 'c043': 7, 'c044': 7, 'c045': 7,
    
    # Laptop/Computer Interaction (8)
    'c046': 8, 'c047': 8, 'c048': 8, 'c049': 8, 'c050': 8, 'c051': 8, 'c052': 8,
    
    # Shoe Interaction (9)
    'c053': 9, 'c054': 9, 'c055': 9, 'c056': 9, 'c057': 9, 'c058': 9,
    
    # Food and Eating (10)
    'c061': 10, 'c062': 10, 'c063': 10, 'c064': 10, 'c065': 10, 'c066': 10, 'c067': 10, 'c068': 10, 'c069': 10, 'c156': 10, 'c147': 10,
    
    # Blanket/Pillow Interaction (11)
    'c070': 11, 'c071': 11, 'c072': 11, 'c073': 11, 'c074': 11, 'c075': 11, 'c076': 11, 'c077': 11, 'c078': 11, 'c079': 11, 'c080': 11,
    
    # Shelf Interaction (12)
    'c081': 12, 'c082': 12,
    
    # Picture/Photo Interaction (13)
    'c083': 13, 'c084': 13, 'c085': 13, 'c086': 13, 'c088': 13,
    
    # Mirror Interaction (14)
    'c093': 14, 'c094': 14, 'c095': 14, 'c096': 14, 'c144': 14,
    
    # Cleaning and Tidying (15)
    'c130': 15,'c098': 15, 'c099': 15, 'c100': 15, 'c101': 15, 'c102': 15, 'c111': 15, 'c121': 15, 'c127': 15, 'c136': 15, 'c137': 15, 'c138': 15, 'c139': 15, 'c118': 15, 'c119': 15, 'c120': 15, 'c126': 15,
    
    # Light Interaction (16)
    'c103': 16, 'c104': 16, 'c105': 16,
    
    # Drinking/Pouring (17)
    'c106': 17, 'c107': 17, 'c108': 17, 'c109': 17, 'c110': 17, 'c128': 17, 'c129': 17,
    
    # Sofa/Bed Interaction (18)
    'c122': 18, 'c123': 18, 'c124': 18, 'c125': 18, 'c133': 18, 'c134': 18, 'c135': 18, 'c146': 18, 'c151': 18,  
    'c154': 18,

    # Emotional Expressions
    'c131': 19, 'c149': 19, 'c152': 19, 'c153': 19,
     
    # Running 
    'c150': 20
}