from multiprocessing import Process, RLock
import time

class TestProcess(Process):
    def __init__(self, container, lock, name):
        self.container = container
        self.lock = lock
        super(TestProcess, self).__init__(name=name)

    def run(self):
        # for _ in range(5):
        while True:
            self.lock.acquire()
            self.container.append(self.name)
            print self.container
            self.lock.release()
            time.sleep(0.1)

container = []
lock = RLock()
processes = [TestProcess(container, lock, str(i)) for i in range(5)]
for process in processes:
    process.start()
