class YoloDataset():
    def __init__(self ,annotation_lines ,input_shape ,num_classes ,train = True):
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.length = len(self.annotation_lines)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return index % self.length
