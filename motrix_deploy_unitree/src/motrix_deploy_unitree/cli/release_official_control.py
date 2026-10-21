# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Interactive release of official robot control."""

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from motrix_deploy_unitree.config.runtime import UnitreeTransportConfig
from motrix_deploy_unitree.go2.control_mode import release_official_control


def run_release_official_control(*, network_interface: str, domain_id: int = 0, timeout_s: float = 5.0) -> int:
    """Release only after explicit confirmation; never create a LowCmd publisher."""
    config = UnitreeTransportConfig(
        network_interface=network_interface, domain_id=domain_id, connect_timeout_s=timeout_s
    )
    print("WARNING: This may stand down the robot and remove built-in balance support.")
    print(
        "Prepare reliable support, clearance and an independent stop. Official control is not restored automatically."
    )
    try:
        confirmed = input("Type 'yes' to release official motion control: ").strip()
    except (KeyboardInterrupt, EOFError):
        confirmed = ""
    if confirmed != "yes":
        print("Cancelled; no hardware connection was opened.")
        return 0
    ChannelFactoryInitialize(config.domain_id, config.network_interface)

    def confirm_stand_down() -> bool:
        print("StandDown was accepted, but that does not prove the movement has finished.")
        try:
            return (
                input("Observe robot down, motion settled and body supported; type 'down' to release: ").strip()
                == "down"
            )
        except (KeyboardInterrupt, EOFError):
            return False

    if not release_official_control(config.connect_timeout_s, confirm_stand_down=confirm_stand_down):
        print("Cancelled after StandDown; motion mode was not released and sport_mode was not disabled.")
        return 1
    print("Official motion mode released; sport_mode disable request succeeded. No LowCmd was sent.")
    return 0
