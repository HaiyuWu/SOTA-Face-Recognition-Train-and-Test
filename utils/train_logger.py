import time
import datetime


class TrainLogger(object):
    def __init__(self, batch_size, frequency, world_size):
        self.batch_size = batch_size
        self.frequency = frequency
        self.world_size = world_size
        self.init = False
        self.tic = 0
        self.last_batch = 0
        self.running_loss = 0

    def __call__(self, epoch, total_epochs, batch, total, loss):
        if self.last_batch > batch:
            self.init = False
        self.last_batch = batch

        if self.init:
            self.running_loss += loss
            if batch % self.frequency == 0:
                speed = self.world_size * self.frequency * self.batch_size / (time.time() - self.tic)
                self.running_loss = self.running_loss / self.frequency
                current_time = datetime.datetime.now().strftime("%d %H:%M")
                log = (
                    f"Time: {current_time} Epoch: [{epoch + 1}-{total_epochs}] Batch: [{batch}-{total}] "
                    + f"Speed: {speed:.2f} samples/sec Loss: {self.running_loss:.5f}"
                )
                print(log)

                self.running_loss = 0
                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()
