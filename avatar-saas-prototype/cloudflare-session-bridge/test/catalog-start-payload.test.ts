import { describe, it, expect } from "vitest";
import { buildInternalStartBody } from "../src/api-v1";

describe("buildInternalStartBody", () => {
  it("translates avatar_id -> avatarId", () => {
    expect(buildInternalStartBody({ avatar_id: "arnold" })).toEqual({
      avatarId: "arnold",
    });
  });

  it("translates bg_id -> bgId", () => {
    expect(buildInternalStartBody({ bg_id: "aloe_window" })).toEqual({
      bgId: "aloe_window",
    });
  });

  it("translates both fields together", () => {
    expect(
      buildInternalStartBody({ avatar_id: "arnold", bg_id: "aloe_window" }),
    ).toEqual({ avatarId: "arnold", bgId: "aloe_window" });
  });

  it("preserves default as default (DO resolves default -> arnold)", () => {
    expect(buildInternalStartBody({ avatar_id: "default" })).toEqual({
      avatarId: "default",
    });
  });

  it("omits undefined avatar_id", () => {
    expect(buildInternalStartBody({ bg_id: "aloe_window" })).toEqual({
      bgId: "aloe_window",
    });
  });

  it("omits undefined bg_id", () => {
    expect(buildInternalStartBody({ avatar_id: "arnold" })).toEqual({
      avatarId: "arnold",
    });
  });

  it("returns empty object when both fields are omitted", () => {
    expect(buildInternalStartBody({})).toEqual({});
  });

  it("serializes to JSON with camelCase keys", () => {
    const body = buildInternalStartBody({
      avatar_id: "arnold",
      bg_id: "aloe_window",
    });
    const json = JSON.stringify(body);
    expect(json).toBe('{"avatarId":"arnold","bgId":"aloe_window"}');
    expect(json).not.toContain("avatar_id");
    expect(json).not.toContain("bg_id");
  });
});
