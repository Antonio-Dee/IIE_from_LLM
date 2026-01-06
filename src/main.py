import json

import knowledge_graph as kg
from model import Model

def extract_information(lm: Model, context: str, verbose: bool = False) -> tuple[set, list, list]:
	# Step 1.1: extract the entities
	entities = lm.extract_entities(context)

	# Step 1.2: extract the grounded explicit relationships (with nesting), then identify whether they are asserted in the text
	relationships = lm.extract_relationships(context, list(entities))

	# Step 1.3: extract the implicit relationships
	inferences = lm.infer_implicit(context, list(entities))

	if verbose:
		print('\n*** INFORMATION EXTRACTION ***\n')
		print('Entities:', str(entities))
		print('Relationships:', str(relationships))
		print('Inferences:', str(inferences))

	return entities, relationships, inferences

def validate_inference(lm: Model, context: str, relationships: list, inferences: list, verbose: bool = False) -> tuple[list, list, list]:
	# Top-level inferences only (nested triplets have not been parsed yet)
	to_validate = [{ 'relationship': inf, 'count': 0 } for inf in inferences]
	validated_triplets = []
	discarded_triplets = []

	if verbose:
		print('\n*** INFERENCE VALIDATION ***\n')

	# Step 2.1: challenge each inferred relationship
	while len(to_validate) > 0:
		item = to_validate.pop(0)
		rel: kg.Relationship = item['relationship']
		count = item['count']

		# Inferences redundant w.r.t. the explicit relationships are discarded right away
		if lm.detect_duplicate(context, relationships, rel, verbose):
			continue

		accepted, explanation = lm.challenge_inference(context, rel, verbose)
		if not accepted:
			# Count the number of times a triplet has been rejected: after 3 rejections, discard it permanently
			count += 1
			if count < 3:
				# Correct the relationship and add the correction, if any, to the set of relationships to validate
				# The correction only regards top-level inferred triplets, potential nested objects are not parsed yet
				corrected_rel = lm.correct_inference(context, rel, explanation, verbose)
				if corrected_rel is not None:
					# If the corrected triplet is a duplicate of any of the validated triplets, it is not validated again
					if not lm.detect_duplicate(context, validated_triplets, corrected_rel, verbose):
						to_validate.append({ 'relationship': corrected_rel, 'count': count })

				rel.discard(reason=explanation, correction=None if corrected_rel is None else str(corrected_rel))
			else:
				rel.discard(reason=explanation)

			discarded_triplets.append(rel)
		else:
			validated_triplets.append(rel)

	if verbose:
		print()

	# Step 2.2: repair the validated relationships to fix the badly formed ones
	# Nested triplets are now parsed and added as objects
	repaired_triplets = validated_triplets

	if verbose:
		print()

	# Step 2.3: for each remaining inferred relationship, extract a grounded premise to explain it
	premises = []
	for triplet in repaired_triplets:
		if not triplet.is_nested:
			premise = lm.explain_inference(context, triplet, relationships, verbose)
		else:
			premises.append([])
		premises.append(premise)

	return repaired_triplets, premises, discarded_triplets

def extract_timing(lm: Model, context: str, relationships: list, verbose: bool = False) -> tuple[list, dict, dict]:
	# Step 3.1: for each triplet, determine whether it is an event or a state and extract a temporal reference if present
	relationships_augmented, residuals = lm.classify_events(context, relationships)
	if verbose:
		print('\n*** TEMPORAL INFORMATION EXTRACTION ***\n')
		for r in relationships_augmented:
			print(f'{r}: {"event" if r.is_event else "state"} with temporal reference {r.time_reference}')

	# Step 3.2: for each pair of triplets, determine the temporal relation between them (before/after/while/none)
	events = [r for r in relationships if r.is_event]
	temporal_relations = lm.order_temporal(context, events)
	if verbose:
		print()
		for k, v in temporal_relations.items():
			if v != '<none>':
				t1, t2 = k
				print(t1, v, t2)

	return relationships_augmented, residuals, temporal_relations

def canonicalize(lm: Model, schema, context: str, relationships: list, verbose: bool = False) -> list:
	"""
	Consolidates the validated relationships within the current schema and returns a list of updated relationships.
	"""

	if verbose:
		print('\n*** SCHEMA CANONICALIZATION ***\n')

	relationships_canonical = []

	for rel in relationships:
		# Step 4.1: extract and vectorize a natural language definition
		definition, definition_vec = lm.define(context, rel)

		if rel.is_nested:
			schema.add_relationship(rel.name, definition, definition_vec)
			relationships_canonical.append(rel)
			if verbose:
				print(f'\n{rel} skipped (nested)')
		else:
			# Step 4.2: find potentially equivalent relationship in the schema via similarity search
			candidates = schema.similarity_search(definition_vec, k=3, threshold=0.2)

			if len(candidates) > 0:
				if verbose:
					print(f'\nRelation: {rel} - {rel.name}: {definition}')
					print(f'Candidates: {candidates}')

				# Step 4.3: check whether replacing the current relationship with one of the candidates is feasible in the context
				rel_canonical = lm.replace(context, rel, definition, candidates)
				print(f'Chosen candidate: {rel_canonical}')
				# Step 4.4: if no substitution was made, update the schema, otherwise update the relationship
				if rel_canonical.lower() == rel.name.lower():
					schema.add_relationship(rel.name, definition, definition_vec)
					if verbose:
						print(f'{rel} unchanged (unsuitable candidates)')
				else:
					rep_old = str(rel)
					rel.update_name(rel_canonical)
					if verbose:
						print(f'{rep_old} changed to {rel}')

				relationships_canonical.append(rel)
			else:
				if verbose:
					print(f'\n{rel} unchanged (no candidates)')
				schema.add_relationship(rel.name, definition, definition_vec)
				relationships_canonical.append(rel)

	return relationships_canonical

def serialize(
		model_type: str,
		index: int,
		context: str,
		entities: list[kg.Entity],
		relationships: list[kg.Relationship],
		inferences: list[kg.Relationship],
		premises: list[list[kg.Relationship]],
		temporal_relations: dict,
		residuals: dict):

	version = 3
	path = f'./out/v{version}/{model_type}/train_{index:05}.json'
	with open(path, 'w') as f:
		f.write('{"context":')
		f.write(f'"{context}","entities":[')
		f.write(','.join([ent.asjson() for ent in entities]))

		f.write('],"relationships":[')
		f.write(','.join([rel.asjson() for rel in relationships]))

		f.write('],"explanations":[')
		f.write(
			','.join(
				[
					'{"premises":[' + ','.join([f'"{clause}"' for clause in premise]) + f'],"conclusion":"{inference}"' + '}'
					for inference, premise in zip(inferences, premises)
					if len(premise) > 0
				]
			)
		)

		f.write('],"temporal_relations":[')
		f.write(
			','.join(
				[
					'{'
					+ f'"eventBefore":"{e2 if v == '<after>' else e1}",'
					+ f'"eventAfter":"{e1 if v == '<after>' else e2}",'
					+ f'"simultaneous":"{str(v == '<while>').lower()}"'
					+ '}'
					for (e1, e2), v in temporal_relations.items()
					if v != '<none>'
				]
			)
		)

		f.write('],"residual_relations":[')
		f.write(
			','.join(
				[
					'{'
					+ f'"triplet":"{rep}",'
					+ f'"details":"{val}"'
					+ '}'
					for rep, val in residuals.items()
				]
			)
		)

		f.write(']}')

def main(model_type: str = 'mistral', num_samples=-1, offset=0, verbose: bool = False):
	# Step 0: load dataset and initialize KG schema, then iterate through the dataset
	lm = Model(model_type)

	with open('./data/SocialIQA/siqa-train.jsonl') as fp:
		for idx, line in enumerate(fp):
			if idx < offset:
				continue
			if num_samples >= 0 and idx >= num_samples + offset:
				break

			sample = json.loads(line)['context']

			if verbose:
				print('\nCONTEXT:', sample)

			# Step 1: information extraction
			entities, relationships, inferences = extract_information(lm, sample, verbose)

			# Step 2: inference validation
			inferences_validated, premises, inferences_discarded = validate_inference(lm, sample, relationships, inferences, verbose)
			# Add the entities introduced by implicit relationships
			entities.update([kg.Entity(name=inf.subject, type='<imp>') for inf in inferences_validated])
			entities.update([kg.Entity(name=inf.object, type='<imp>') for inf in inferences_validated if inf.object is not None and not inf.has_nesting])
			if verbose:
				print(f'\nUpdated entities: {entities}')

			# Step 3: temporal information extraction
			valid_relationships, residuals, temporal_relations = extract_timing(lm, sample, relationships + inferences_validated, verbose)

			# Step 4: schema canonicalization
			relationships_canonical = valid_relationships

			# Step 5: serialize the results
			serialize(model_type, idx, sample, entities, relationships_canonical + inferences_discarded, inferences_validated, premises, temporal_relations, residuals)

if __name__ == '__main__':
	main('mistral', num_samples=15, offset=0, verbose=True)