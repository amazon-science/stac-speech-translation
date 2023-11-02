#!/usr/bin/env/python3
""" Script to evaluate turn detection with F1-score

Authors
 * Paturi, Rohit 2023
 * Juan Zuluaga-Gomez, 2023
"""

import argparse
import copy
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


@dataclass
class Segment:
    start_time: float
    end_time: float
    speakers: List[str]
    overlap: bool = False

    def duration(self):
        return self.end_time - self.start_time


@dataclass
class Overlap:
    ref_idx: int
    ref_overlap_proportion: float
    pred_overlap_proportion: float
    overlap_start: float
    overlap_end: float


@dataclass
class Interval:
    start_time: float
    end_time: float

    def duration(self):
        return self.end_time - self.start_time

    def __hash__(self):
        return hash((self.start_time, self.end_time))

    def is_in_range(self, val):
        return True if self.start_time <= val and val < self.end_time else False


def read_rttm(path):
    rttm_file = open(path)
    segmentation = []
    for line in rttm_file:
        splitted = line.strip().split(" ")
        name = splitted[1]
        start = float(splitted[3])
        duration = float(splitted[4])
        speaker = splitted[7]
        segment = (name, start, duration, speaker)
        segmentation.append(segment)
    rttm_file.close()
    return segmentation


def get_ref_hyp_overlaps(
    pred_segments: List[Segment], ref_segments: List[Segment]
) -> List[List[Overlap]]:
    all_overlaps = []
    for pred_segment in pred_segments:
        curr_overlaps = []
        for ref_idx, ref_segment in enumerate(ref_segments):
            # If opposite is true then definitely there is no overlap.
            if (pred_segment.start_time <= ref_segment.end_time) and (
                pred_segment.end_time >= ref_segment.start_time
            ):
                overlap_start_time = max(
                    pred_segment.start_time, ref_segment.start_time
                )
                overlap_end_time = min(pred_segment.end_time, ref_segment.end_time)

                ref_overlap_proportion = 0
                if ref_segment.duration() > 0:
                    ref_overlap_proportion = (
                        overlap_end_time - overlap_start_time
                    ) / ref_segment.duration()
                pred_overlap_proportion = 0
                if pred_segment.duration() > 0:
                    pred_overlap_proportion = (
                        overlap_end_time - overlap_start_time
                    ) / pred_segment.duration()

                curr_overlaps.append(
                    Overlap(
                        ref_idx,
                        ref_overlap_proportion,
                        pred_overlap_proportion,
                        overlap_start_time,
                        overlap_end_time,
                    )
                )
        all_overlaps.append(curr_overlaps)
    return all_overlaps


def get_segments_from(segmentation, ignore_overlap=True):
    """parse the input from RTTM into a list with Segment class"""

    segments = []
    prev_segment = None
    for name, start_time, duration, spk_label in segmentation:
        # TODO: Mark overlapping speech when creating test set.
        #  This way with an optional flag we can compute all metrics by either considering or not considering overlapping speech.
        # For now, this code assumes no overlapping speech.
        cur_segment = Segment(
            start_time=start_time, end_time=start_time + duration, speakers=[spk_label]
        )
        if ignore_overlap:
            if prev_segment is not None:
                # if there is overlap between segments
                # 3 cases are possible: 1) full overlap 2) partial overlap 3)no overlap
                # overlap cases
                if cur_segment.start_time < prev_segment.end_time:
                    # partial overlap
                    if cur_segment.end_time > prev_segment.end_time:
                        # adjust prev_segment end_time to avoid overlap
                        segments[-1].end_time = cur_segment.start_time
                        cur_segment.start_time = prev_segment.end_time
                        segments.append(copy.copy(cur_segment))
                        prev_segment = cur_segment
                    # full overlap. Discard current segment
                    elif cur_segment.end_time < prev_segment.end_time:
                        # TODO: split up the previous segment
                        pass
                else:
                    segments.append(copy.copy(cur_segment))
                    prev_segment = cur_segment
            else:
                segments.append(copy.copy(cur_segment))
                prev_segment = cur_segment

        else:
            segments.append(copy.copy(cur_segment))

    return segments


def get_segments_from_v2(alignment_output):
    segments = []
    for o in alignment_output:
        segments.append(
            Segment(
                start_time=o["start_time"],
                end_time=o["end_time"],
                speakers=[o["character"]],
            )
        )
    return segments


def get_rttm_line(segment: Segment, pred_speaker: str, pred_to_ref_speaker: dict):
    """

    Parameters
    ----------
    interval
    pred_speaker
    pred_to_ref_speaker

    Returns
    -------

    """
    return f"SPEAKER name 1 {round(segment.start_time, 4)} {round(segment.duration(), 4)} <NA> <NA> {pred_to_ref_speaker[pred_speaker]} <NA> <NA>"


def store_rttm_file(rttm_output: List[str], rttm_output_file_path: str):
    """

    Parameters
    ----------
    rttm_output
    rttm_output_file_path

    Returns
    -------

    """
    with open(rttm_output_file_path, "w") as f:
        for line in rttm_output:
            f.write(line + "\n")


def get_total_audio_duration(ref_rttm_file_path: str) -> float:
    with open(ref_rttm_file_path, "r") as f:
        last_line = f.readlines()[-1]
        last_line_parts = last_line.strip().split()
        ref_end_time = float(last_line_parts[3]) + float(last_line_parts[4])
        return ref_end_time


def get_speaker_change_time_ranges(
    segmentation_ref, tolerance: float = 0.1
) -> List[float]:
    ref_segments: List[Segment] = get_segments_from(segmentation_ref)
    acceptable_change_intervals: List[Interval] = []

    current_start = ref_segments[0].start_time
    current_speakers = []
    cnt = 0
    for ref_idx, ref_segment in enumerate(ref_segments):
        ref_segment.speakers[
            0
        ] = f"SPK-{cnt}"  # add a name to set each turn as a different speaker
        if set(ref_segment.speakers) != set(current_speakers):
            new_start_time = current_start - tolerance
            if len(acceptable_change_intervals) >= 1:
                prev_interval = acceptable_change_intervals[-1]
                new_start_time = (
                    prev_interval.end_time
                    if (current_start - tolerance) < prev_interval.end_time
                    else new_start_time
                )
            acceptable_change_intervals.append(
                Interval(new_start_time, ref_segment.start_time + tolerance)
            )
        current_speakers = ref_segment.speakers
        current_start = ref_segment.end_time
        cnt += 1

    unacceptable_change_intervals: List[float] = []
    total_audio_duration = get_total_audio_duration(ref_rttm_file_path)

    if len(acceptable_change_intervals) > 0:
        if acceptable_change_intervals[0].start_time > 0:
            unacceptable_change_intervals.append(
                Interval(0, acceptable_change_intervals[0].start_time)
            )

    for i in range(1, len(acceptable_change_intervals)):
        unacc_interval = Interval(
            acceptable_change_intervals[i - 1].end_time,
            acceptable_change_intervals[i].start_time,
        )
        # don't add interval if duration=0
        if unacc_interval.duration() > 0:
            unacceptable_change_intervals.append(unacc_interval)

    if len(acceptable_change_intervals) > 0:
        if acceptable_change_intervals[-1].end_time < total_audio_duration:
            unacceptable_change_intervals.append(
                Interval(acceptable_change_intervals[-1].end_time, total_audio_duration)
            )

    return acceptable_change_intervals, unacceptable_change_intervals


def get_predicted_speaker_change_times(segmentation_hyp):
    """function that gets the points where the model predicted a speaker change"""

    pred_segments: List[Segment] = get_segments_from(segmentation_hyp)

    current_speakers = []
    predicted_speaker_change_times = []
    cnt = 0
    for segment in pred_segments:
        segment.speakers[0] = f"SPK-{cnt}"
        if set(segment.speakers) != set(current_speakers):
            predicted_speaker_change_times.append(segment.start_time)
        current_speakers = segment.speakers
        cnt += 1
    return predicted_speaker_change_times


def get_ref_change_interval_to_predicted_change_time(
    change_intervals: List[Interval], predicted_speaker_change_times: List[float]
) -> Dict[Interval, float]:
    """Function that outputs the actual change intervals from ref file"""

    change_interval_to_predicted_change_time = dict()
    for change_interval in change_intervals:
        for change_time in predicted_speaker_change_times:
            if (
                change_interval.start_time <= change_time
                and change_time < change_interval.end_time
            ):
                if change_interval in change_interval_to_predicted_change_time:
                    change_interval_to_predicted_change_time[change_interval].append(
                        change_time
                    )
                else:
                    change_interval_to_predicted_change_time[change_interval] = [
                        change_time
                    ]
            # No need to search beyond this range
            if change_time >= change_interval.end_time:
                break
    return change_interval_to_predicted_change_time


def get_speaker_change_detection_metrics(
    segmentation_ref, segmentation_hyp, tolerance=0.1
):
    (
        acceptable_change_intervals,
        unacceptable_change_intervals,
    ) = get_speaker_change_time_ranges(segmentation_ref, tolerance=tolerance)
    predicted_speaker_change_times = get_predicted_speaker_change_times(
        segmentation_hyp
    )

    acceptable_change_interval_to_predicted_change_time = (
        get_ref_change_interval_to_predicted_change_time(
            acceptable_change_intervals, predicted_speaker_change_times
        )
    )
    unacceptable_change_interval_to_predicted_change_time = (
        get_ref_change_interval_to_predicted_change_time(
            unacceptable_change_intervals, predicted_speaker_change_times
        )
    )

    # print(f"len(acc dict) = {len(acceptable_change_interval_to_predicted_change_time)}")
    # print(f"len(unacc dict) = {len(unacceptable_change_interval_to_predicted_change_time)}")

    n_acc, n_unacc = 0, 0
    n_acc_lg1, n_unacc_lg1 = 0, 0
    for change_interval in acceptable_change_interval_to_predicted_change_time:
        n_acc += len(
            acceptable_change_interval_to_predicted_change_time[change_interval]
        )
        if (
            len(acceptable_change_interval_to_predicted_change_time[change_interval])
            > 1
        ):
            # print(change_interval, acceptable_change_interval_to_predicted_change_time[change_interval])
            n_acc_lg1 += 1

    for change_interval in unacceptable_change_interval_to_predicted_change_time:
        n_unacc += len(
            unacceptable_change_interval_to_predicted_change_time[change_interval]
        )
        if (
            len(unacceptable_change_interval_to_predicted_change_time[change_interval])
            > 1
        ):
            # print(change_interval, unacceptable_change_interval_to_predicted_change_time[change_interval])
            n_unacc_lg1 += 1
    # ipdb.set_trace()
    # print(f"len(acc dict) = {len(acceptable_change_interval_to_predicted_change_time)}")
    # print(f"len(unacc dict) = {len(unacceptable_change_interval_to_predicted_change_time)}")
    # print(f"n_acc_lg1 = {n_acc_lg1}, n_unacc_lg1 = {n_unacc_lg1}")
    # print(f"n_acc = {n_acc}, n_unacc = {n_unacc}, n_acc + n_unacc = {n_acc + n_unacc}")
    # print(f"len(pred ch times) = {len(predicted_speaker_change_times)}")
    # import ipdb; ipdb.set_trace()

    tp = len(acceptable_change_interval_to_predicted_change_time)
    fn = len(acceptable_change_intervals) - tp
    tn = len(unacceptable_change_interval_to_predicted_change_time)

    fp_multiple_same_interval = n_acc - tp
    fp = fp_multiple_same_interval + n_unacc

    # print(f"tp + fp = {tp + fp}")
    # print(f"tp + fn = {tp + fn}")
    # assert (tp + fp)  == len(predicted_speaker_change_times)
    # # Following assertion is trivially true.
    # assert (tp + fn) == len(acceptable_change_intervals)
    precision = tp / (tp + fp) * 100
    recall = tp / (tp + fn) * 100
    f1 = 2 / (1 / precision + 1 / recall)

    miss_detection = fn / (fn + tp) * 100
    f_pos_rate = fp / (fp + tn) * 100
    return precision, recall, f1, miss_detection, f_pos_rate


def get_speaker_label_accuracy(segmentation_ref, updated_segmentation_hyp):
    pred_segments: List[Segment] = get_segments_from(updated_segmentation_hyp)
    ref_segments: List[Segment] = get_segments_from(segmentation_ref)

    # Switch ref and pred arguments. To get perspective from reference standpoint.
    # Note: In this case pred_overlap_proportion is actually ref_overlap_proportion and vice-versa.
    all_overlaps: List[List[Overlap]] = get_ref_hyp_overlaps(
        pred_segments, ref_segments
    )

    labels = []
    predicted_speaker_labels = []
    for ref_idx, overlaps in enumerate(all_overlaps):
        max_overlap, max_overlap_idx = -1, -1
        for j, v in enumerate(overlaps):
            ref_overlap_prop = v.pred_overlap_proportion
            if ref_overlap_prop > max_overlap:
                max_overlap = ref_overlap_prop
                max_overlap_idx = j
        if max_overlap_idx >= 0:
            pred_segment_idx = overlaps[max_overlap_idx].ref_idx
            prediction = "+".join(pred_segments[pred_segment_idx].speakers)
            predicted_speaker_labels.append(prediction)
            actual = "+".join(ref_segments[ref_idx].speakers)
            labels.append(1 if prediction == actual else 0)
        # no overlapping segments for current reference segment.
        else:
            labels.append(0)
            predicted_speaker_labels.append("0")
    speaker_label_accuracy = sum(labels) / len(ref_segments) * 100
    return speaker_label_accuracy, predicted_speaker_labels


def assign_characters(segmentation_hyp, segmentation_ref):
    pred_segments: List[Segment] = get_segments_from(segmentation_hyp)
    ref_segments: List[Segment] = get_segments_from(segmentation_ref)

    all_overlaps: List[List[Overlap]] = get_ref_hyp_overlaps(
        pred_segments, ref_segments
    )
    # print("Assigning predicted speaker labels to reference speaker labels")
    #
    # print("Getting the cost matrix for Hungarian algorithm...")
    # count how many seconds ref speaker label and predicted speaker label overlap
    speaker_matches = {}  # {ref_speaker: {pred_speaker: seconds}}
    pred_speakers_set = set()
    for i, (pred_segment, curr_overlaps) in enumerate(zip(pred_segments, all_overlaps)):
        for overlap in curr_overlaps:
            total_time = overlap.overlap_end - overlap.overlap_start
            for ref_speaker in ref_segments[overlap.ref_idx].speakers:
                if ref_speaker not in speaker_matches:
                    speaker_matches[ref_speaker] = {}
                for pred_speaker in pred_segment.speakers:
                    pred_speakers_set.add(pred_speaker)
                    if pred_speaker not in speaker_matches[ref_speaker]:
                        speaker_matches[ref_speaker][pred_speaker] = 0
                    speaker_matches[ref_speaker][pred_speaker] += total_time
    max_speaker = max(speaker_matches, key=lambda s: sum(speaker_matches[s].values()))

    # use Hungarian algorithm to assign ref speaker label to predicted speaker label based on total number of overlapping seconds
    # if more predicted speakers than ref speakers, do multiple rounds until all predicted speakers are assigned
    # print("Applying Hungarian algorithm to obtain results...")
    ref_speakers = list(speaker_matches.keys())
    pred_speakers = list(pred_speakers_set)
    pred_to_ref_speaker = {}  # {pred_speaker: ref_speaker}
    while len(pred_speakers) > 0:
        to_remove = set()
        matches = np.array(
            [
                [speaker_matches[s1].get(s2, 0) for s2 in pred_speakers]
                for s1 in ref_speakers
            ]
        )
        ref_speaker_idxs, pred_speaker_idxs = linear_sum_assignment(matches)
        for ref_speaker_i, pred_speaker_i in zip(ref_speaker_idxs, pred_speaker_idxs):
            ref_speaker = ref_speakers[ref_speaker_i]
            pred_speaker = pred_speakers[pred_speaker_i]
            to_remove.add(pred_speaker)
            pred_to_ref_speaker[pred_speaker] = ref_speaker
        for pred_speaker in to_remove:
            pred_speakers.remove(pred_speaker)

    # print("Resulting map--")
    # for i, pred_spk in enumerate(pred_to_ref_speaker):
    #     print(f"pred_spk = {pred_spk}, ref_spk = {pred_to_ref_speaker[pred_spk]}")

    updated_hyp_rttm_file_path = hyp_rttm_file_path.split(".rttm")[0] + "_updated.rttm"
    rttm_output = []
    for segment in pred_segments:
        rttm_output.append(
            get_rttm_line(segment, segment.speakers[0], pred_to_ref_speaker)
        )

    store_rttm_file(rttm_output, updated_hyp_rttm_file_path)

    return updated_hyp_rttm_file_path


def prune_hyp_seg_from_ref_seg(segmentation_hyp, segmentation_ref):
    """prune hyp rttm to be within ref time ranges"""

    start_hyp_index = 0
    end_hyp_index = len(segmentation_hyp)
    for i, hyp_seg in enumerate(segmentation_hyp):
        if hyp_seg[1] > segmentation_ref[0][1] or (
            hyp_seg[1] < segmentation_ref[0][1]
            and hyp_seg[1] + hyp_seg[2] > segmentation_ref[0][1]
        ):
            start_hyp_index = i
            break
    for i, hyp_seg in enumerate(segmentation_hyp):
        if (
            hyp_seg[1] + hyp_seg[2] > segmentation_ref[-1][1] + segmentation_ref[-1][2]
            and hyp_seg[1] < segmentation_ref[-1][1] + segmentation_ref[-1][2]
        ) or (hyp_seg[1] > segmentation_ref[-1][1] + segmentation_ref[-1][2]):
            end_hyp_index = i
            break
    segmentation_hyp = segmentation_hyp[
        start_hyp_index : min(end_hyp_index + 1, len(segmentation_hyp))
    ]

    return segmentation_hyp


def evaluate_speaker_turn_detection_davidhzc(
    ref_rttm_file_path: str = None,
    hyp_rttm_file_path: str = None,
    tolerances: List[float] = [0.25],
    merge_overlap_gt_regions: bool = True,
    calclate_mean_turn_point_in_time: bool = False,
):
    """this function tailors for the RTTM that Pablo shsared, which assumes speaker changes adjacent segments in the reference"""
    segmentation_ref = read_rttm(ref_rttm_file_path)
    segmentation_hyp = read_rttm(hyp_rttm_file_path)

    df_ref = pd.DataFrame(
        segmentation_ref, columns=["utt_id", "start_time", "duration", "speaker_label"]
    )
    df_hyp = pd.DataFrame(
        segmentation_hyp, columns=["utt_id", "start_time", "duration", "speaker_label"]
    )
    df_ref["end_time"] = df_ref["start_time"] + df_ref["duration"]
    df_hyp["end_time"] = df_hyp["start_time"] + df_hyp["duration"]

    # if it is the raw output for speaker diarization
    list_spkr = list(set(df_hyp["speaker_label"]))
    if len(list_spkr) > 1:
        df_hyp["group"] = (
            df_hyp["speaker_label"] != df_hyp["speaker_label"].shift(1)
        ).cumsum()
        group_to_spkr_mapping = pd.Series(
            df_hyp["speaker_label"].tolist(), index=df_hyp["group"].tolist()
        ).to_dict()
        df_hyp_merged = (
            df_hyp.iloc[1:, :]
            .groupby("group")
            .agg({"start_time": "min", "end_time": "max"})
        )
        df_hyp_merged["speaker_label"] = [
            group_to_spkr_mapping[i] for i in df_hyp_merged.index
        ]
        hyp_turns = pd.DataFrame(
            {
                "turn_start": df_hyp_merged["end_time"][:-1].tolist(),
                "turn_end": df_hyp_merged["start_time"][1:].tolist(),
            }
        ).round(3)
        hyp_turns["overlap"] = hyp_turns["turn_start"] > hyp_turns["turn_end"]
        hyp_turns["hyp_turn_region"] = hyp_turns.apply(
            lambda x: sorted([x["turn_start"], x["turn_end"]]), axis=1
        )

        if calclate_mean_turn_point_in_time:
            hyp_turns["start_time"] = (
                hyp_turns["turn_start"] + hyp_turns["turn_end"]
            ) / 2
            hyp_turns["end_time"] = (
                hyp_turns["turn_start"] + hyp_turns["turn_end"]
            ) / 2
        else:
            hyp_turns["start_time"] = hyp_turns["hyp_turn_region"].apply(lambda x: x[0])
            hyp_turns["end_time"] = hyp_turns["hyp_turn_region"].apply(lambda x: x[1])
        df_hyp = hyp_turns.copy()

    ref_turns = pd.DataFrame(
        {
            "turn_start": df_ref["end_time"][:-1].tolist(),
            "turn_end": df_ref["start_time"][1:].tolist(),
        }
    ).round(3)
    ref_turns["overlap"] = ref_turns["turn_start"] > ref_turns["turn_end"]
    ref_turns["turn_region"] = ref_turns.apply(
        lambda x: sorted([x["turn_start"], x["turn_end"]]), axis=1
    )

    num_gt_change_points_unmerged = df_ref.shape[0] - 1
    num_predicted_change_points = df_hyp.shape[0]

    for tolerance in tolerances:
        ref_turns["turn_region_w_tol"] = ref_turns["turn_region"].apply(
            lambda x: [x[0] - tolerance, x[1] + tolerance]
        )
        ref_turns[["ref_start", "ref_end"]] = np.array(
            ref_turns["turn_region_w_tol"].tolist()
        )

        if merge_overlap_gt_regions:
            # merge overlap regions
            ref_turns["group"] = (
                ref_turns["ref_start"] > ref_turns["ref_end"].shift()
            ).cumsum()
            ref_turn_merged = ref_turns.groupby("group").agg(
                {"ref_start": "min", "ref_end": "max"}
            )
            gt_intervals = pd.arrays.IntervalArray.from_arrays(
                ref_turn_merged["ref_start"], ref_turn_merged["ref_end"], closed="both"
            )
        else:
            # keep the original reference despite some overlaps
            gt_intervals = pd.arrays.IntervalArray.from_arrays(
                ref_turns["ref_start"], ref_turns["ref_end"], closed="both"
            )

        # evaluate turn detection
        pred_correct, pred_incorrect = 0, 0
        detected_turns = []
        for predicted_cp_start, predicted_cp_end in df_hyp[
            ["start_time", "end_time"]
        ].values.tolist():
            overlap_ind = gt_intervals.overlaps(
                pd.Interval(predicted_cp_start, predicted_cp_end)
            )
            detected_turns.append([i for i, x in enumerate(overlap_ind) if x])
            if sum(overlap_ind) >= 1:
                pred_correct += 1
            else:
                pred_incorrect += 1

        fp = pred_incorrect
        tp = pred_correct
        num_gt_change_points = len(gt_intervals)

        # miss detection
        detected_change_points_correct = set(sum(detected_turns, []))
        num_miss = num_gt_change_points - len(detected_change_points_correct)
        miss_detection_rate = num_miss / num_gt_change_points * 100

        # false alarm
        false_alarm_rate = fp / num_predicted_change_points * 100

        # recall, precision, f1
        recall = len(detected_change_points_correct) / num_gt_change_points * 100
        precision = tp / num_predicted_change_points * 100
        f1 = 2 / (1 / precision + 1 / recall)

        print(
            f"#speaker changes (original ref rttm): {num_gt_change_points_unmerged}; "
            f"\n#speaker changes (predicted hyp rttm): {num_predicted_change_points};"
            f"\n#merged speaker change regions with tolerance={tolerance}s: {len(gt_intervals)}"
        )
        print(f"Tolerance | Precision | Recall | F1-score | Miss | FA")
        print(
            f"{tolerance}s: {precision:.2f} {recall:.2f} {f1:.2f} {miss_detection_rate:.2f} {false_alarm_rate:.2f}"
        )
        print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--ref_rttm",
        type=str,
        action="store",
        default="",
        help="reference rttm per file",
    )
    parser.add_argument(
        "-s",
        "--hyp_rttm",
        type=str,
        action="store",
        default="",
        help="hypothesis rttm per file",
    )
    parser.add_argument(
        "-t",
        "--tolerance",
        type=float,
        action="store",
        default=0.25,
        help="tolerance around speaker changes in seconds",
    )

    args = parser.parse_args()
    hyp_rttm_file_path = args.hyp_rttm
    ref_rttm_file_path = args.ref_rttm

    tolerances = [float(args.tolerance)]

    evaluate_speaker_turn_detection_davidhzc(
        ref_rttm_file_path=ref_rttm_file_path,
        hyp_rttm_file_path=hyp_rttm_file_path,
        tolerances=tolerances,
        merge_overlap_gt_regions=True,
        calclate_mean_turn_point_in_time=False,
    )
    print("\ndone")
