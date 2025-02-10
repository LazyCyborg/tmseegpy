import re
from dataclasses import dataclass
from typing import List, Dict
import os


@dataclass
class SessionStats:
    session_name: str
    total_events: int
    bad_channels: List[str]
    bad_epochs: List[int]
    first_ica_components: List[int]
    second_ica_components: List[int]
    artifacts_cleaned: int
    artifact_detection_rate: float
    pci_value: float


def parse_log_file(file_path: str) -> List[SessionStats]:
    """Parse a TMS-EEG preprocessing log file and extract session statistics."""
    with open(file_path, 'r') as f:
        content = f.read()

    # Split content into sessions
    sessions = re.split(r'Starting epoched processing of Session \d+: ', content)[1:]

    session_stats = []

    for session in sessions:
        # Extract session name
        session_name = session.split('\n')[0].strip()

        # Extract total events
        events_match = re.search(r'(\d+) events found on stim channel', session)
        total_events = int(events_match.group(1)) if events_match else 0

        # Extract bad channels
        bad_channels_match = re.search(r'Detected bad channels: \[(.*?)\]', session)
        bad_channels = [ch.strip("' ") for ch in bad_channels_match.group(1).split(',')] if bad_channels_match else []

        # Extract bad epochs
        bad_epochs_match = re.search(r'Dropped \d+ epochs: ([\d, ]+)', session)
        bad_epochs = [int(ep) for ep in bad_epochs_match.group(1).split(', ')] if bad_epochs_match else []

        # Extract first ICA components
        first_ica_match = re.search(r'Excluding (\d+) manually selected components: \[([\d, ]+)\]', session)
        first_ica_components = [int(comp) for comp in first_ica_match.group(2).split(', ')] if first_ica_match else []

        # Extract second ICA components
        second_ica_match = re.search(r'Second ICA.*?Excluding (\d+) manually selected components: \[([\d, ]+)\]',
                                     session, re.DOTALL)
        second_ica_components = [int(comp) for comp in
                                 second_ica_match.group(2).split(', ')] if second_ica_match else []

        # Extract artifact cleaning stats
        artifacts_match = re.search(r'Found (\d+) artifacts\nDetection rate: ([\d.]+)%', session)
        artifacts_cleaned = int(artifacts_match.group(1)) if artifacts_match else 0
        artifact_detection_rate = float(artifacts_match.group(2)) if artifacts_match else 0.0

        # Extract PCI value
        pci_match = re.search(r'PCI: ([\d.]+)', session)
        pci_value = float(pci_match.group(1)) if pci_match else 0.0

        session_stats.append(SessionStats(
            session_name=session_name,
            total_events=total_events,
            bad_channels=bad_channels,
            bad_epochs=bad_epochs,
            first_ica_components=first_ica_components,
            second_ica_components=second_ica_components,
            artifacts_cleaned=artifacts_cleaned,
            artifact_detection_rate=artifact_detection_rate,
            pci_value=pci_value
        ))

    return session_stats


def generate_report(sessions: List[SessionStats], output_file: str):
    """Generate a formatted report from session statistics."""
    with open(output_file, 'w') as f:
        f.write("TMS-EEG Preprocessing Report\n")
        f.write("===========================\n\n")

        for i, session in enumerate(sessions, 1):
            f.write(f"Session {i}: {session.session_name}\n")
            f.write("-" * (len(f"Session {i}: {session.session_name}") + 1) + "\n")

            f.write(f"Total Events: {session.total_events}\n")

            f.write("\nBad Channels Removed:\n")
            f.write(f"- Count: {len(session.bad_channels)}\n")
            f.write(f"- Channels: {', '.join(session.bad_channels)}\n")

            f.write("\nBad Epochs Removed:\n")
            f.write(f"- Count: {len(session.bad_epochs)}\n")
            f.write(f"- Epochs: {', '.join(map(str, session.bad_epochs))}\n")

            f.write("\nICA Components:\n")
            f.write(
                f"- First ICA removed: {len(session.first_ica_components)} components {session.first_ica_components}\n")
            f.write(
                f"- Second ICA removed: {len(session.second_ica_components)} components {session.second_ica_components}\n")

            f.write("\nMuscle Artifact Cleaning:\n")
            f.write(f"- Artifacts cleaned: {session.artifacts_cleaned}\n")
            f.write(f"- Detection rate: {session.artifact_detection_rate:.1f}%\n")

            f.write(f"\nPCI Value: {session.pci_value:.2f}\n")

            f.write("\n" + "=" * 50 + "\n\n")

        # Summary statistics
        f.write("Overall Summary\n")
        f.write("--------------\n")
        f.write(f"Total sessions processed: {len(sessions)}\n")
        f.write(f"Average PCI value: {sum(s.pci_value for s in sessions) / len(sessions):.2f}\n")
        f.write(
            f"Average artifact detection rate: {sum(s.artifact_detection_rate for s in sessions) / len(sessions):.1f}%\n")


def main():
    """Command-line interface for the script."""
    import argparse
    parser = argparse.ArgumentParser(description='Generate TMS-EEG preprocessing report from log file')
    parser.add_argument('log_file', help='Path to the preprocessing log file')
    parser.add_argument('--output', '-o', default='preprocessing_report.txt',
                        help='Output file path for the report (default: preprocessing_report.txt)')

    args = parser.parse_args()

    try:
        sessions = parse_log_file(args.log_file)
        generate_report(sessions, args.output)
        print(f"Report successfully generated: {args.output}")
    except Exception as e:
        print(f"Error generating report: {str(e)}")


if __name__ == '__main__':
    main()