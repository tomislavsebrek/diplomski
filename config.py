class Data:

    def __init__(self):
        self.CLASS_NAMES = ["chimeric", "left_repeat", "right_repeat", "normal"]
        self.NUM_CLASSES = len(self.CLASS_NAMES)
        self.PATH = "dataset"
        self.INTERPOLATION_LENGTH = 500
        self.FORMATTED_DATA = "formatted_data/interpolation_" + str(self.INTERPOLATION_LENGTH)
        self.BATCH_SIZE = 32
        self.LATENT_SIZE = 10

        self.PERIODIC_SAVE = 0
        self.EVALUATE_EVERY = 5000
        self.SAVE_LOCATION = "save/checkpointtt.ckpt"
        self.LOAD_LOCATION = "save/checkpointtt.ckpt"
        self.MAX_ITER = 5000000

        self.USE_LABELED_AFTER = 0

        self.USE_TSNE = False
        self.USE_RECONSTRUCTION = True
        self.USE_VISUALIZATION = False
        self.USE_CONFUSION = True
        self.TSNE_COMPONENTS = 2

CONFIG = Data()