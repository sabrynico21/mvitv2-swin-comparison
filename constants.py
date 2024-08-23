
import torch
import torch.nn.functional as F
from torchvision import transforms
from transforms import ResizeVideoTransform, VideoToTensorTransform

PREPROCESSING_TRANSFORMS = transforms.Compose([
    ResizeVideoTransform((224, 224)),
    VideoToTensorTransform()
])
NUM_EPOCHS = 4
NUM_CLASSES = 21
# NO_ACTION_TENSOR = F.one_hot(torch.tensor(NUM_CLASSES-1), num_classes=NUM_CLASSES)
BATCH_SIZE = 16
LEARNING_RATE = 0.001
GRADIENT_ACCUMULATION_ITERS = 4
PRINT_BATCH_LOSS_EVERY = 4
VIDEO_MAX_SAMPLES = 1
SAMPLE_CLIP_SIZE = 16

CLASS_MAPPING = {
    # Clothing and Dressing (0)
    '0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '148': 0, '155': 0,
    
    # Door/Window Interaction (1)
    '6': 1, '7': 1, '8': 1, '97': 1, '89': 1, '90': 1, '91': 1, '92': 1, '112': 1, '113': 1, '140': 1, '141': 1, '142': 1, '143': 1,
    
    # Table Interaction (2)
    '9': 2, '10': 2, '11': 2, '12': 2, '13': 2, '14': 2,
    
    # Phone/Camera Interaction (3)
    '15': 3, '16': 3, '17': 3, '18': 3, '19': 3, '87': 3, '132': 3,
    
    # Bag Handling (4)
    '20': 4, '21': 4, '22': 4, '23': 4, '24': 4,
    
    # Book/Reading (5)
    '25': 5, '26': 5, '27': 5, '28': 5, '29': 5, '30': 5, '31': 5, '32': 5, '145': 5, '115': 5, '116': 5, '117': 5,
    
    # Towel Handling (6)
    '33': 6, '34': 6, '35': 6, '36': 6, '37': 6, '38': 6,
    
    # Box Interaction (7)
    '39': 7, '40': 7, '41': 7, '42': 7, '43': 7, '44': 7, '45': 7,
    
    # Laptop/Computer Interaction (8)
    '46': 8, '47': 8, '48': 8, '49': 8, '50': 8, '51': 8, '52': 8,
    
    # Shoe Interaction (9)
    '53': 9, '54': 9, '55': 9, '56': 9, '57': 9, '58': 9,
    
    # Food and Eating (10)
    '61': 10, '62': 10, '63': 10, '64': 10, '65': 10, '66': 10, '67': 10, '68': 10, '69': 10, '156': 10, '147': 10,
    
    # Blanket/Pillow Interaction (11)
    '70': 11, '71': 11, '72': 11, '73': 11, '74': 11, '75': 11, '76': 11, '77': 11, '78': 11, '79': 11, '80': 11,
    
    # Shelf Interaction (12)
    '81': 12, '82': 12,
    
    # Picture/Photo Interaction (13)
    '83': 13, '84': 13, '85': 13, '86': 13, '88': 13,
    
    # Mirror Interaction (14)
    '93': 14, '94': 14, '95': 14, '96': 14, '144': 14,
    
    # Cleaning and Tidying (15)
    '130': 15, '98': 15, '99': 15, '100': 15, '101': 15, '102': 15, '111': 15, '114': 15, '121': 15, '127': 15, '136': 15, '137': 15, '138': 15, '139': 15, '118': 15, '119': 15, '120': 15, '126': 15,
    
    # Light Interaction (16)
    '103': 16, '104': 16, '105': 16,
    
    # Drinking/Pouring (17)
    '106': 17, '107': 17, '108': 17, '109': 17, '110': 17, '128': 17, '129': 17,
    
    # Sofa/Bed Interaction (18)
    '59': 18, '60': 18, '122': 18, '123': 18, '124': 18, '125': 18, '133': 18, '134': 18, '135': 18, '146': 18, '151': 18, '154': 18,
    
    # Emotional Expressions (19)
    '131': 19, '149': 19, '152': 19, '153': 19,
    
    # Running (20)
    '150': 20
}