
required_class_index = [2, 3, 5, 7, 0]
required_class_names = []

classes_path = "coco.names"
with open(classes_path, "r") as f:
    classNames = f.read().strip().split("\n")

for i in required_class_index:
    required_class_names.append(classNames[i])
    
print(required_class_names)