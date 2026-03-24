
import traceback

try:
    from .qwen3_sensevoice_xy_v4_51_3 import Qwen3MTPSenseVoiceConfig as Qwen3SenseVoiceXYConfig
    from .qwen3_sensevoice_xy_v4_51_3 import Qwen3MTPSenseVoiceForCausalLM as Qwen3SenseVoiceXYForCausalLM

except Exception as error:
    print("=" * 100)
    print(f"{error=}")
    print(traceback.format_exc())
    print("=" * 100)

    pass

try:
    from .vita_utu_mtp_sensevoice_xy_v4_51_3 import UTUV1Config
    from .vita_utu_mtp_sensevoice_xy_v4_51_3 import VITAUTUV1ForCausalLM

except Exception as error:
    print("=" * 100)
    print(f"{error=}")
    print(traceback.format_exc())
    print("=" * 100)

    pass
