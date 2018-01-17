from threading import Thread, RLock
import time

class TestThread(Thread):
    def __init__(self, container, lock, name):
        self.container = container
        self.lock = lock
        super(TestThread, self).__init__(name=name)

    def run(self):
        while True:
            self.lock.acquire()
            self.container.append(self.getName())
            print self.container
            self.lock.release()
            time.sleep(0.1)

container = []
lock = RLock()
threads = [TestThread(container, lock, str(i)) for i in range(5)]
for thread in threads:
    thread.start()
