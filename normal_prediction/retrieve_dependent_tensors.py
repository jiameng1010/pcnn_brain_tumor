import tensorflow as tf

class Tensor_Retrieval:
    def __init__(self, graph, tensor, place_holder):
        self.graph = graph
        self.tensor = tensor
        self.place_holder = place_holder

        self.op = self.find_op(self.tensor)
        self.dependent_tensors = []
        self.create_dep(tensor)

    def get_dep(self):
        return self.dependent_tensors

    def find_op(self, tensor):

        n=0
        for op in self.graph.get_operations():
            if tensor in op.outputs:
                n += 1
        if n != 1:
            print('wirld')

        for op in self.graph.get_operations():
            if tensor in op.outputs:
                return op
        return None

    def is_depend(self, tensor1, tensor2):
        for tensor in self.find_op(tensor1).inputs:
            if tensor == tensor2:
                return True
            if self.is_depend(tensor, tensor2):
                return True
        return False

    def create_dep(self, target):
        dependent_tensors = []
        for tensor in self.find_op(target).inputs:
            if not tensor == self.place_holder:
                if not self.is_depend(tensor, self.place_holder):
                    if tensor not in self.dependent_tensors:
                        self.dependent_tensors.append(tensor)
                else:
                    self.create_dep(tensor)
