import time
import logging


class TrainLogger(object):
    def __init__(self, batch_size, frequency, steps, total_epoch, writer):
        self.batch_size = batch_size
        self.total_epochs = total_epoch
        self.total_steps = steps
        self.frequency = frequency
        self.time_start = time.time()
        self.start_step = 0
        self.init = False
        self.tic = 0
        self.writer = writer

    def __call__(self,
                 step,
                 epoch,
                 loss,
                 local_rank
                 ):
        if local_rank == 0 and step > 0 and step % self.frequency == 0:
            if self.init:
                try:
                    speed_total: float = self.frequency * self.batch_size / (time.time() - self.tic)
                except ZeroDivisionError:
                    speed_total = float('inf')

                time_now = time.time()
                time_sec = int(time_now - self.time_start)
                time_sec_avg = time_sec / (step - self.start_step + 1)
                eta_sec = time_sec_avg * (self.total_steps - step - 1)
                time_for_end = eta_sec/3600
                if self.writer is not None:
                    self.writer.add_scalar('time_for_end', time_for_end, step)
                    self.writer.add_scalar('loss', loss.avg, step)
                msg = "Epoch: [%d-%d]  Speed %.2f samples/sec   Loss %.4f   Global Step: %d   Required: %1.f hours"\
                      % (epoch, self.total_epochs, speed_total, loss.avg, step, time_for_end)
                logging.info(msg)
                loss.reset()
                self.tic = time.time()
            else:
                self.init = True
                self.tic = time.time()
