import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from rdflib import Graph, Literal, Namespace
from rdflib.namespace import RDF, XSD


ONTOLOGY_NS = "http://www.semanticweb.org/rafab/ontologies/PIBITI2526#"


FEATURE_CLASS_MAP = {
	"web": "Web",
	"boss": "Boss",
	"chamfer": "Chamfer",
	"corner": "Corner",
	"flange": "Flange",
	"hole": "Hole",
	"slot": "Slot",
	"thread": "Thread",
	"attachment hole": "AttachmentHole",
	"tooling hole": "ToolingHole",
	"attachment flange": "AttachmentFlange",
	"deformed flange": "DeformedFlange",
	"stiffening flange": "StiffeningFlange",
}

REQUIRED_JSON_KEYS = {
	"part_name",
	"thickness",
	"holes",
	"bosses",
	"slots",
	"chamfers",
	"threads",
	"flanges",
	"corners",
	"total_features",
	"method",
	"improved_multi_agent",
}


@dataclass
class FeatureRecord:
	label: str
	feature_id: int
	parent_feature_id: Optional[int]
	position: Optional[Tuple[float, float, float]]
	normal: Optional[Tuple[float, float, float]]
	hole_diameter: Optional[float]
	corner_radius: Optional[float]
	flange_width: Optional[float]
	flange_length: Optional[float]
	flange_bend_radius: Optional[float]
	flange_type_raw: Optional[str]
	flange_type_direction: Optional[str]
	flange_type_multiplicity: Optional[str]
	flange_type_placement: Optional[str]
	flange_type_geometry: Optional[str]
	flange_type_angle_type: Optional[str]


class PairValidationError(Exception):
	pass


def safe_identifier(raw: str) -> str:
	cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", raw.strip())
	cleaned = re.sub(r"_+", "_", cleaned)
	return cleaned.strip("_") or "unnamed"


def parse_mm_float(text: str) -> float:
	match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
	if not match:
		raise ValueError(f"Nao foi possivel converter valor numerico: {text}")
	return float(match.group(0))


def parse_point_tuple(text: str) -> Tuple[float, float, float]:
	match = re.search(
		r"\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*"
		r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*"
		r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)",
		text,
	)
	if not match:
		raise ValueError(f"Nao foi possivel converter tupla 3D: {text}")
	return float(match.group(1)), float(match.group(2)), float(match.group(3))


def split_flange_type(raw_type: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
	parts = [part.strip() for part in raw_type.split(",") if part.strip()]
	while len(parts) < 5:
		parts.append(None)
	return parts[0], parts[1], parts[2], parts[3], parts[4]


def parse_feature_line(line: str) -> FeatureRecord:
	line_match = re.match(r"^(?P<label>.+?)\s*\((?P<content>.+)\)\s*$", line.strip())
	if not line_match:
		raise ValueError(f"Linha de feature invalida: {line}")

	label = line_match.group("label").strip()
	content = line_match.group("content")

	pairs: Dict[str, str] = {}
	for chunk in [part.strip() for part in content.split(";") if part.strip()]:
		if ":" not in chunk:
			continue
		key, value = chunk.split(":", 1)
		pairs[key.strip().lower()] = value.strip()

	if "id" not in pairs:
		raise ValueError(f"Feature sem ID: {line}")

	parent_id = int(pairs["parent id"]) if "parent id" in pairs else None

	flange_type_raw = pairs.get("type")
	flange_direction = None
	flange_multiplicity = None
	flange_placement = None
	flange_geometry = None
	flange_angle_type = None
	if flange_type_raw:
		(
			flange_direction,
			flange_multiplicity,
			flange_placement,
			flange_geometry,
			flange_angle_type,
		) = split_flange_type(flange_type_raw)

	return FeatureRecord(
		label=label,
		feature_id=int(pairs["id"]),
		parent_feature_id=parent_id,
		position=parse_point_tuple(pairs["position point"]) if "position point" in pairs else None,
		normal=parse_point_tuple(pairs["position normal"]) if "position normal" in pairs else None,
		hole_diameter=parse_mm_float(pairs["diameter"]) if "diameter" in pairs else None,
		corner_radius=parse_mm_float(pairs["radius"]) if "radius" in pairs else None,
		flange_width=parse_mm_float(pairs["width"]) if "width" in pairs else None,
		flange_length=parse_mm_float(pairs["length"]) if "length" in pairs else None,
		flange_bend_radius=parse_mm_float(pairs["bend radius"]) if "bend radius" in pairs else None,
		flange_type_raw=flange_type_raw,
		flange_type_direction=flange_direction,
		flange_type_multiplicity=flange_multiplicity,
		flange_type_placement=flange_placement,
		flange_type_geometry=flange_geometry,
		flange_type_angle_type=flange_angle_type,
	)


def parse_txt_file(txt_path: Path) -> Tuple[str, float, List[FeatureRecord]]:
	lines = txt_path.read_text(encoding="utf-8").splitlines()
	if len(lines) < 2:
		raise PairValidationError(f"Arquivo TXT incompleto: {txt_path}")

	part_match = re.match(r"^Part name:\s*(.+?)\s*$", lines[0].strip())
	thick_match = re.match(r"^Part thickness is:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*mm\s*$", lines[1].strip())
	if not part_match or not thick_match:
		raise PairValidationError(f"Cabecalho TXT invalido em {txt_path}")

	part_name = part_match.group(1).strip()
	thickness = float(thick_match.group(1))

	features: List[FeatureRecord] = []
	for raw in lines[2:]:
		clean = raw.strip()
		if not clean:
			continue
		features.append(parse_feature_line(clean))

	if not features:
		raise PairValidationError(f"Nenhuma feature encontrada em {txt_path}")

	return part_name, thickness, features


def validate_json_file(json_path: Path) -> dict:
	try:
		data = json.loads(json_path.read_text(encoding="utf-8"))
	except json.JSONDecodeError as exc:
		raise PairValidationError(f"JSON invalido em {json_path}: {exc}") from exc

	missing = REQUIRED_JSON_KEYS.difference(data.keys())
	if missing:
		missing_display = ", ".join(sorted(missing))
		raise PairValidationError(f"JSON sem chaves obrigatorias em {json_path}: {missing_display}")
	return data


def pair_basename_from_any(name: str) -> str:
	base = name.strip()
	if base.lower().endswith(".txt"):
		base = base[:-4]
	if base.lower().endswith("_extraction.json"):
		base = base[:-16]
	return base


def discover_pairs(input_dir: Path) -> Dict[str, Tuple[Path, Path]]:
	pairs: Dict[str, Tuple[Path, Path]] = {}
	for txt_path in sorted(input_dir.glob("*.txt")):
		base = txt_path.stem
		json_path = input_dir / f"{base}_extraction.json"
		if json_path.exists():
			pairs[base] = (txt_path, json_path)
	return pairs


def select_pairs(
	all_pairs: Dict[str, Tuple[Path, Path]],
	selected_names: Optional[Iterable[str]],
) -> Dict[str, Tuple[Path, Path]]:
	if not selected_names:
		return all_pairs

	normalized_selection = {pair_basename_from_any(name) for name in selected_names}
	not_found = sorted(name for name in normalized_selection if name not in all_pairs)
	if not_found:
		missing_display = ", ".join(not_found)
		raise PairValidationError(f"Arquivos selecionados nao encontrados: {missing_display}")

	return {key: all_pairs[key] for key in sorted(normalized_selection)}


def ensure_pair_consistency(
	txt_part_name: str,
	txt_thickness: float,
	json_data: dict,
	txt_path: Path,
	json_path: Path,
) -> None:
	if txt_part_name != str(json_data.get("part_name", "")).strip():
		raise PairValidationError(
			f"Divergencia de part_name entre {txt_path.name} e {json_path.name}: "
			f"TXT='{txt_part_name}' JSON='{json_data.get('part_name')}'"
		)

	json_thickness = float(json_data.get("thickness"))
	if abs(txt_thickness - json_thickness) > 1e-9:
		raise PairValidationError(
			f"Divergencia de thickness entre {txt_path.name} e {json_path.name}: "
			f"TXT={txt_thickness} JSON={json_thickness}"
		)


def class_for_label(label: str) -> str:
	return FEATURE_CLASS_MAP.get(label.lower().strip(), "MachiningFeature")


def add_string(graph: Graph, subject, predicate, value: Optional[str]) -> None:
	if value is not None and value != "":
		graph.add((subject, predicate, Literal(value, datatype=XSD.string)))


def add_int(graph: Graph, subject, predicate, value: Optional[int]) -> None:
	if value is not None:
		graph.add((subject, predicate, Literal(int(value), datatype=XSD.integer)))


def add_double(graph: Graph, subject, predicate, value: Optional[float]) -> None:
	if value is not None:
		graph.add((subject, predicate, Literal(float(value), datatype=XSD.double)))


def pair_already_added(graph: Graph, base_name: str) -> bool:
	ontology = Namespace(ONTOLOGY_NS)
	base_token = safe_identifier(base_name)
	new_workpiece_uri = ontology[base_token]
	legacy_workpiece_uri = ontology[f"workpiece_{base_token}"]
	return (
		(new_workpiece_uri, RDF.type, ontology.Workpiece) in graph
		or (legacy_workpiece_uri, RDF.type, ontology.Workpiece) in graph
	)


def process_pair(graph: Graph, base_name: str, txt_path: Path, json_path: Path) -> Tuple[int, int]:
	json_data = validate_json_file(json_path)
	txt_part_name, txt_thickness, features = parse_txt_file(txt_path)
	ensure_pair_consistency(txt_part_name, txt_thickness, json_data, txt_path, json_path)

	ontology = Namespace(ONTOLOGY_NS)
	base_token = safe_identifier(base_name)

	workpiece_uri = ontology[base_token]
	graph.add((workpiece_uri, RDF.type, ontology.Workpiece))
	add_string(graph, workpiece_uri, ontology.hasPartName, txt_part_name)
	add_double(graph, workpiece_uri, ontology.hasThickness, txt_thickness)

	feature_uri_map = {}
	for feature in features:
		feature_id_token = f"{feature.feature_id:02d}"
		feature_uri = ontology[f"{base_token}_{feature_id_token}"]
		feature_uri_map[feature.feature_id] = feature_uri

		feature_class = class_for_label(feature.label)
		graph.add((feature_uri, RDF.type, ontology[feature_class]))
		graph.add((workpiece_uri, ontology.hasFeature, feature_uri))
		graph.add((feature_uri, ontology.isFeatureOf, workpiece_uri))

		add_int(graph, feature_uri, ontology.hasID, feature.feature_id)
		add_int(graph, feature_uri, ontology.hasParentID, feature.parent_feature_id)

		if feature.position:
			add_double(graph, feature_uri, ontology.hasPositionPointX, feature.position[0])
			add_double(graph, feature_uri, ontology.hasPositionPointY, feature.position[1])
			add_double(graph, feature_uri, ontology.hasPositionPointZ, feature.position[2])
		if feature.normal:
			add_double(graph, feature_uri, ontology.hasPositionNormalX, feature.normal[0])
			add_double(graph, feature_uri, ontology.hasPositionNormalY, feature.normal[1])
			add_double(graph, feature_uri, ontology.hasPositionNormalZ, feature.normal[2])

		add_double(graph, feature_uri, ontology.hasDiameter, feature.hole_diameter)
		add_double(graph, feature_uri, ontology.hasRadius, feature.corner_radius)
		add_double(graph, feature_uri, ontology.hasWidth, feature.flange_width)
		add_double(graph, feature_uri, ontology.hasLength, feature.flange_length)
		add_double(graph, feature_uri, ontology.hasBendRadius, feature.flange_bend_radius)

		add_string(graph, feature_uri, ontology.hasType, feature.flange_type_raw)

	for feature in features:
		if feature.parent_feature_id is None:
			continue
		child_uri = feature_uri_map[feature.feature_id]
		parent_uri = feature_uri_map.get(feature.parent_feature_id)
		if parent_uri is not None:
			graph.add((child_uri, ontology.hasParentFeature, parent_uri))
			graph.add((parent_uri, ontology.hasChildFeature, child_uri))

	return len(features), 1


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description=(
			"Cria individuos de Workpiece/MachiningFeature no PIBITI2526.rdf "
			"a partir de pares TXT + _extraction.json."
		)
	)
	parser.add_argument(
		"--input-dir",
		default="test_afr_improved_multi_agent_extractor",
		help="Diretorio com os pares TXT/JSON.",
	)
	parser.add_argument(
		"--rdf-path",
		default="PIBITI2526.rdf",
		help="Arquivo RDF OWL que recebera os individuos.",
	)
	parser.add_argument(
		"--select",
		nargs="+",
		help=(
			"Lista de basenames para processar (1 ou mais). Aceita nome puro, .txt "
			"ou _extraction.json."
		),
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Valida e parseia os pares sem gravar no RDF.",
	)
	return parser


def main() -> int:
	parser = build_argument_parser()
	args = parser.parse_args()

	input_dir = Path(args.input_dir)
	rdf_path = Path(args.rdf_path)

	if not input_dir.exists() or not input_dir.is_dir():
		parser.error(f"Diretorio de entrada invalido: {input_dir}")
	if not rdf_path.exists() or not rdf_path.is_file():
		parser.error(f"Arquivo RDF nao encontrado: {rdf_path}")

	pairs = discover_pairs(input_dir)
	if not pairs:
		parser.error(f"Nenhum par TXT/JSON encontrado em: {input_dir}")

	try:
		selected_pairs = select_pairs(pairs, args.select)
	except PairValidationError as exc:
		parser.error(str(exc))

	graph = Graph()
	graph.parse(rdf_path, format="xml")

	already_added_pairs = []
	pending_pairs = {}
	for base_name, pair in selected_pairs.items():
		if pair_already_added(graph, base_name):
			already_added_pairs.append(base_name)
		else:
			pending_pairs[base_name] = pair

	total_pairs = len(selected_pairs)
	successful_pairs = 0
	failed_pairs = 0
	total_features_created = 0
	skipped_pairs = len(already_added_pairs)

	for base_name, (txt_path, json_path) in pending_pairs.items():
		try:
			features_count, _ = process_pair(graph, base_name, txt_path, json_path)
			successful_pairs += 1
			total_features_created += features_count
			print(f"[OK] {base_name}: {features_count} features processadas")
		except Exception as exc:
			failed_pairs += 1
			print(f"[ERRO] {base_name}: {exc}")

	if not args.dry_run and successful_pairs > 0:
		graph.serialize(destination=str(rdf_path), format="xml")

	mode = "DRY-RUN" if args.dry_run else "GRAVACAO"
	print("\nResumo")
	print(f"- Modo: {mode}")
	print(f"- Pares selecionados: {total_pairs}")
	print(f"- Pares ja adicionados (pulados): {skipped_pairs}")
	print(f"- Pares pendentes para insercao: {len(pending_pairs)}")
	print(f"- Pares com sucesso: {successful_pairs}")
	print(f"- Pares com erro: {failed_pairs}")
	print(f"- Features processadas: {total_features_created}")
	if already_added_pairs:
		print(f"- Lista pulada: {', '.join(already_added_pairs)}")
	if args.dry_run:
		print("- RDF nao foi alterado (dry-run).")
	elif successful_pairs == 0:
		print("- Nenhum par valido: RDF nao foi alterado.")
	else:
		print(f"- RDF atualizado em: {rdf_path}")

	return 0 if failed_pairs == 0 else 2


if __name__ == "__main__":
	raise SystemExit(main())
