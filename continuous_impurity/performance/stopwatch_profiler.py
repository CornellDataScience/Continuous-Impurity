import timeit

class StopwatchProfiler:
    #don't assume this is error-free, but very likely is
    def __init__(self):
        self.__start_stop_intervals = []
        self.__last_start_time = None
        self.__lap_click_times = []

    def start(self):
        if self.__is_running():
            raise ValueError("Attempted to start stopwatch while it was already running")
        self.__last_start_time = timeit.default_timer()

    def stop(self):
        if not self.__is_running():
            raise ValueError("Attempted to stop stopwatch when it is not running")
        stop_time = timeit.default_timer()
        self.__start_stop_intervals.append((self.__last_start_time, stop_time))
        self.__last_start_time = None


    def reset(self):
        self.__last_start_time = None
        self.__lap_click_times = []
        self.__start_stop_intervals = []


    def lap(self, name):
        if not self.__is_running():
            raise ValueError("Attempted to lap before starting the stopwatch")
        self.__lap_click_times.append((name, timeit.default_timer()))

    def elapsed_time(self):
        out = 0
        for start_stop in self.__start_stop_intervals:
            out += start_stop[1] - start_stop[0]
        return out

    def lap_deltas(self):
        out = []
        prev_lap_click_time = self.__start_stop_intervals[0][0]
        for lap_click in self.__lap_click_times:
            out.append((lap_click[0], self.__running_time_in_interval(prev_lap_click_time, lap_click[1])))
            prev_lap_click_time = lap_click[1]
        return out

    def __running_time_in_interval(self, start, stop):
        curr_start_stop_ind = self.__start_stop_that_contains(start)
        if self.__start_stop_contains(self.__start_stop_intervals[curr_start_stop_ind], stop):
            return stop - start

        out = self.__start_stop_intervals[curr_start_stop_ind][1] - start
        curr_start_stop_ind += 1

        while curr_start_stop_ind < len(self.__start_stop_intervals):
            if self.__start_stop_contains(self.__start_stop_intervals[curr_start_stop_ind], stop):
                out += stop - self.__start_stop_intervals[curr_start_stop_ind][0]
                return out
            else:
                out += self.__start_stop_intervals[curr_start_stop_ind][1] - \
                    self.__start_stop_intervals[curr_start_stop_ind][0]
            curr_start_stop_ind += 1
        return None


    def __start_stop_that_contains(self, time):
        for i in range(len(self.__start_stop_intervals)):
            start_stop = self.__start_stop_intervals[i]
            if self.__start_stop_contains(start_stop, time):
                return i
        return None

    def __start_stop_contains(self, start_stop, time):
        return time >= start_stop[0] and time <= start_stop[1]

    def relative_lap_deltas(self):
        out = self.lap_deltas()
        tot_time = self.elapsed_time()
        for i in range(len(out)):
            out[i] = (out[i][0], out[i][1]/tot_time)
        return out


    def __is_running(self):
        return self.__last_start_time is not None

    def __str__(self):
        out = ""
        out += "Total elapsed time: " + str(self.elapsed_time()) + ", "
        out += "Lap times: " + str(self.lap_deltas())
        return out

    def __repr__(self):
        return str(self)
