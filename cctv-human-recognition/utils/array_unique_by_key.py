def array_unique_by_key(items: list[dict[str, object]], key: str):
    seen = set()
    unique_items = []

    for item in items:
        value = item.get(key)
        if value not in seen:
            unique_items.append(item)
            seen.add(value)

    return unique_items