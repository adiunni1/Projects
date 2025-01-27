"""Appointments project
Adi Unni
This file contains an appointment class,
as well as an agenda class"""
from datetime import datetime

class Appt:
    """An appointment has a start time, an end time, and a title.
    The start and end time should be on the same day.
    """

    def __init__(self, start: datetime, finish: datetime, desc: str):
        assert finish > start, f"Period finish ({finish}) must be after start ({start})"
        self.start = start
        self.finish = finish
        self.desc = desc

    def __lt__(self, other: 'Appt') -> bool:
        """Determines whether this appointment ends before or right when the other starts"""
        return self.finish <= other.start

    def __gt__(self, other: 'Appt') -> bool:
        """Determines whether this appointment starts after or when the other ends"""
        return self.start >= other.finish

    def __eq__(self, other: 'Appt') -> bool:
        """Equality means same time period, ignoring description"""
        return self.start == other.start and self.finish == other.finish

    def overlaps(self, other: 'Appt') -> bool:
        """Is there a non-zero overlap between these periods?"""
        return not (self < other or self > other)

    def intersect(self, other: 'Appt') -> 'Appt':
        """The overlapping portion of two Appt objects"""
        if self.overlaps(other):
            start_time = max(self.start, other.start)
            finish_time = min(self.finish, other.finish)
            return Appt(start_time, finish_time, "Overlap")
        else:
            return None

    def __str__(self) -> str:
        """The textual format of an appointment"""
        date_iso = self.start.date().isoformat()
        start_iso = self.start.time().isoformat(timespec='minutes')
        finish_iso = self.finish.time().isoformat(timespec='minutes')
        return f"{date_iso} {start_iso} {finish_iso} | {self.desc}"


class Agenda:
    """A collection of appointments"""

    def __init__(self):
        self.elements = []

    def __len__(self) -> int:
        """Number of appointments in the agenda"""
        return len(self.elements)

    def __str__(self):
        """Each appointment on its own line"""
        return "\n".join(str(e) for e in self.elements)

    def __repr__(self) -> str:
        """Representation for debugging purposes"""
        return f"Agenda({self.elements})"

    def sort(self):
        """Sort agenda by appointment start times"""
        self.elements.sort(key=lambda appt: appt.start)

    def conflicts(self) -> 'Agenda':
        """Returns an agenda consisting of the conflicts"""
        self.sort()
        conflicts_agenda = Agenda()

        for i, appt1 in enumerate(self.elements):
            for appt2 in self.elements[i + 1:]:
                if appt1.overlaps(appt2):
                    conflicts_agenda.append(appt1.intersect(appt2))
                elif appt2.start > appt1.finish:
                    break  # No further conflicts possible
        return conflicts_agenda

    def append(self, appt: Appt):
        """Add an appointment to the agenda"""
        self.elements.append(appt)


if __name__ == "__main__":
    appt1 = Appt(datetime(2024, 3, 15, 13, 30), datetime(2024, 3, 15, 15, 30), "Early afternoon nap")
    appt2 = Appt(datetime(2024, 3, 15, 15, 00), datetime(2024, 3, 15, 16, 00), "Coffee break")

    agenda = Agenda()
    agenda.append(appt1)
    agenda.append(appt2)

    ag_conflicts = agenda.conflicts()
    print(f"In agenda:\n{agenda}")
    print(f"Conflicts:\n{ag_conflicts}")


