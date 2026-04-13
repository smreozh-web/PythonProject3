class CircularBuffer:
    def __init__(self, size):
        self.size = size
        self.data = []

    def add(self, value):
        self.data.append(value)
        if len(self.data) > self.size:
            self.data.pop(0)

    def get(self):
        return self.data

    def avg(self):
        return sum(self.data)/len(self.data) if self.data else 0