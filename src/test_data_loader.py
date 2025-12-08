from data_loader import get_data_generators

train, val, test = get_data_generators(
    "data/train",
    "data/val",
    "data/test"
)

print("Train batches:", len(train))
print("Val batches:", len(val))
print("Test batches:", len(test))
print("Classes:", train.class_indices)
