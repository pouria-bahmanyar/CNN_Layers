from torch.utils.data import DataLoader

def data_loader(dataset):
  loader = DataLoader(dataset, batch_size = 64, num_workers = 2)
  return loader
