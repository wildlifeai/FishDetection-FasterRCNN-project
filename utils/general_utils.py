
def collate_fn(batch):
    """
    :param batch: The current batch
    :return: Zip the batch into a tuples
    """
    return tuple(zip(*batch))
