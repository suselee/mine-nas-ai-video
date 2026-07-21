#include <assert.h>
#include <stdio.h>

#include "track_fusion.h"

using namespace dw;

static IvaResult one_person(uint32_t id, float x, float height) {
    IvaResult result;
    IvaObject person = {};
    person.id = id;
    person.score = 0.9f;
    person.x1 = x;
    person.x2 = x + 0.2f;
    person.y1 = 0.4f;
    person.y2 = person.y1 + height;
    result.people.push_back(person);
    return result;
}

int main() {
    FusionConfig config = {};
    config.probable_min_seconds = 4;
    config.probable_min_observations = 3;
    config.child_max_height_ratio = 0.55;
    config.relative_child_height_ratio = 0.75;
    config.face_check_interval_seconds = 2;
    config.confirmed_ttl_seconds = 8;
    config.track_lost_seconds = 3;
    config.mqtt_update_seconds = 5;
    config.face_threshold = 0.35f;
    config.face_high_threshold = 0.55f;
    TrackFusion fusion(config);

    fusion.observe(0, one_person(7, 0.1f, 0.35f));
    assert(fusion.has_due_face_candidate(0));
    fusion.mark_due_face_candidates_checked(0);
    assert(!fusion.has_due_face_candidate(1));
    assert(fusion.has_due_face_candidate(2));
    fusion.observe(1, one_person(17, 0.11f, 0.35f));
    fusion.observe(2, one_person(27, 0.12f, 0.35f));
    fusion.observe(4, one_person(37, 0.13f, 0.35f));
    std::vector<FusionEvent> events = fusion.collect_events(4);
    assert(events.size() == 1 && events[0].event == "start");
    assert(events[0].identity == "probable");
    uint32_t logical_track = events[0].track_id;

    fusion.apply_face_score(logical_track, 0.60f, 5);
    events = fusion.collect_events(5);
    assert(events.size() == 1 && events[0].event == "update");
    assert(events[0].identity == "confirmed");

    TrackFusion repeated(config);
    repeated.observe(0, one_person(1, 0.1f, 0.35f));
    repeated.apply_face_score(1, 0.40f, 1);
    assert(repeated.collect_events(1).empty());
    repeated.apply_face_score(1, 0.40f, 2);
    events = repeated.collect_events(2);
    assert(events.size() == 1 && events[0].identity == "confirmed");

    TrackFusion expired(config);
    expired.observe(0, one_person(1, 0.1f, 0.35f));
    expired.apply_face_score(1, 0.40f, 1);
    expired.apply_face_score(1, 0.40f, 5);
    assert(expired.collect_events(5).empty());

    fusion.observe(6, IvaResult());
    events = fusion.collect_events(9);
    assert(events.size() == 1 && events[0].event == "end");
    assert(events[0].identity == "confirmed");

    printf("TRACK_FUSION_TEST_OK\n");
    return 0;
}
