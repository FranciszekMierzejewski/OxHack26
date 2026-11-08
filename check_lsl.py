from pylsl import StreamInlet, resolve_streams

streams = resolve_streams(wait_time=5)
print(f"Found {len(streams)} LSL stream(s)")
for stream in streams:
    inlet = StreamInlet(stream)
    info = inlet.info()
    print(f"\nname={info.name()} type={info.type()} rate={info.nominal_srate()} channels={info.channel_count()}")
    channel = info.desc().child("channels").child("channel")
    print("labels:", [(channel.child_value("label"), channel := channel.next_sibling())[0] for _ in range(info.channel_count())])
    sample, _ = inlet.pull_sample(timeout=5)
    print("sample:", None if sample is None else [round(v, 2) for v in sample])