"""复用SuperMimic S54的灰阶配色，为原始USD生成独立外观图层。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SOURCE = "supermimic_s54/tools/render.py:apply_render_style"
# 来源：相邻SuperMimic工程的POLICY_RGBA / POLICY_JOINT_RGBA / POLICY_DARK_RGBA。
# 颜色与部位分区保持一致；采用非金属、较高粗糙度，接近参考中的低高光风格。
# 将数值保存在本项目中，生成和回放均不依赖另一个工程是否存在。
PALETTE = {
    "ShellGray": ((0.78, 0.78, 0.78), 0.0, 0.80),
    "JointGray": ((0.30, 0.30, 0.30), 0.0, 0.80),
    "EndDark": ((0.10, 0.10, 0.10), 0.0, 0.80),
}


def usd_object(value: Any, description: str) -> Any:
    """检查USD对象；pxr的部分C++绑定缺少准确返回类型，在此收敛动态类型边界。"""
    if not value:
        raise RuntimeError(f"无法创建或读取USD对象：{description}")
    return value


def bind_material(prim, material, strength=UsdShade.Tokens.fallbackStrength):
    binding = usd_object(UsdShade.MaterialBindingAPI.Apply(prim), "材质绑定")
    binding.Bind(material, strength)


def srgb_to_linear(color):
    """USD材质存线性RGB；Newton会转换为显示用sRGB，不能直接把色号写进去。"""
    return tuple(c / 12.92 if c <= .04045 else ((c + .055) / 1.055) ** 2.4 for c in color)


def make_material(stage, root, name):
    srgb, metallic, roughness = PALETTE[name]
    material = usd_object(UsdShade.Material.Define(stage, root + "/Looks/S54_" + name), name)
    shader = usd_object(UsdShade.Shader.Define(stage, str(material.GetPath()) + "/Shader"), name + "着色器")
    # 当前Isaac Lab/Newton的颜色兼容层读取OmniPBR的diffuse_color_constant。
    # 只写UsdPreviewSurface.diffuseColor会被该兼容层跳过，newton_gl可能仍显示白色。
    shader.CreateImplementationSourceAttr().Set(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*srgb_to_linear(srgb)))
    shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(roughness)
    output = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(output)
    material.CreateSurfaceOutput("mdl").ConnectToSource(output)
    return material


def mesh_material(name):
    """按参考render.py的网格名称分区，避免凭关节编号猜测外壳部位。"""
    if name.endswith(("foot_roll", "hand_pitch_nohand")) or name in ("head_camera", "head_rader"):
        return "EndDark"
    if name == "waist_yaw" or name.endswith((
        "leg_roll", "leg_yaw", "foot_pitch", "arm_roll", "arm_yaw", "hand_yaw", "hand_roll",
    )):
        return "JointGray"
    return "ShellGray"


def generate(source: Path, output: Path):
    source, output = source.resolve(), output.resolve()
    if source == output:
        raise ValueError("外观图层必须与原始USD分开保存")
    if not source.is_file():
        raise FileNotFoundError(source)
    original = usd_object(Usd.Stage.Open(str(source)), str(source))
    if not original.GetDefaultPrim():
        raise ValueError("原始USD没有defaultPrim")
    stage = usd_object(Usd.Stage.CreateInMemory(), "外观图层")
    stage.GetRootLayer().subLayerPaths = [str(source)]
    # defaultPrim和单位是stage级元数据，需要显式复制到新的根图层。
    root_prim = stage.GetPrimAtPath(original.GetDefaultPrim().GetPath())
    root = str(root_prim.GetPath())
    stage.SetDefaultPrim(root_prim)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(original))
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(original))
    stage.GetRootLayer().documentation = (
        "S54 visual appearance only. Grayscale palette and mesh assignment adapted from "
        + REFERENCE_SOURCE)
    materials = {name: make_material(stage, root, name) for name in PALETTE}
    count = 0
    for link in root_prim.GetChildren():
        visuals = link.GetChild("visuals")
        if not visuals:
            continue
        # 保留全部visual实例和原始头部网格；不再拆分或添加黑色面罩。
        mesh_names = [prim.GetName() for prim in Usd.PrimRange(visuals, Usd.TraverseInstanceProxies())
                      if prim.IsA(UsdGeom.Mesh)]
        material_names = {mesh_material(name) for name in mesh_names}
        if len(material_names) != 1:
            raise ValueError(f"{visuals.GetPath()}需要单独处理材质分区：{mesh_names}")
        # 较强的祖先绑定覆盖原STL统一白色材质，不解除实例共享。
        material = materials[material_names.pop()]
        bind_material(visuals, material, UsdShade.Tokens.strongerThanDescendants)
        count += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    text = stage.GetRootLayer().ExportToString()
    text = text.replace("@" + str(source) + "@", "@" + os.path.relpath(source, output.parent) + "@")
    output.write_text(text)
    print(f"SuperMimic灰阶USD已生成：{output}；着色visual组={count}，新增网格=0")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "assets/biped_s54.usd")
    parser.add_argument("--output", type=Path, default=ROOT / "assets/biped_s54_colored.usda")
    args = parser.parse_args()
    generate(args.source, args.output)


if __name__ == "__main__":
    main()
