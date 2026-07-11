from nas_video_summarizer.llm import AnalysisResult


def _result(*, keep=False, confidence=0.85, title="", summary="", tags=None):
    return AnalysisResult(
        keep=keep,
        title=title,
        summary=summary,
        tags=tags or [],
        confidence=confidence,
        start_offset_seconds=0,
        end_offset_seconds=1,
        raw={},
    )


def test_repairs_high_confidence_child_activity_with_false_keep():
    result = _result(
        title="Child playing",
        summary="A young child is engaged in a quiet activity at home.",
        tags=["child", "play"],
    )

    assert result.keep_consistency_repaired(0.5) is True
    assert result.should_save(0.5) is True


def test_does_not_repair_scene_without_child_evidence():
    result = _result(
        title="Quiet Study Scene",
        summary="A desk and workspace in an empty room.",
        tags=["desk", "study"],
    )

    assert result.keep_consistency_repaired(0.5) is False
    assert result.should_save(0.5) is False


def test_does_not_repair_explicit_exclusion_or_low_confidence():
    sleeping = _result(
        summary="A young child is sleeping after an activity.",
        tags=["child"],
    )
    uncertain = _result(
        confidence=0.6,
        summary="A young child is playing.",
        tags=["child", "play"],
    )
    idle = _result(
        summary="A young child is not engaged in any activity.",
        tags=["child", "activity"],
    )

    assert sleeping.should_save(0.5) is False
    assert uncertain.should_save(0.5) is False
    assert idle.should_save(0.5) is False


def test_original_true_keep_behavior_is_unchanged():
    assert _result(keep=True, confidence=0.8).should_save(0.5) is True
    assert _result(keep=True, confidence=0.3).should_save(0.5) is False
