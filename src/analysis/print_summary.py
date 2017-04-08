import argparse
import json
import os

_SUMMARY_JSON = 'summary.json'

_VOC_VERSIONS = ['VOC2006', 'VOC2007']

_VOC_CLASSES = {
	_VOC_VERSIONS[0]: ['bicycle', 'bus', 'car', 'cat', 'cow', 'dog', 'horse',
      'motorbike', 'person', 'sheep'],
	_VOC_VERSIONS[1]: ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
      'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
      'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],
}

def LoadSummaryJsons(params):
	summary_jsons = {}
	for result_dir in params['result_dirs']:
		summary_jsons[result_dir] = json.load(open(
			os.path.join(result_dir, _SUMMARY_JSON), 'r'))
	return summary_jsons

def PrintAveragePrecisionHeader(params):
	for cls in _VOC_CLASSES[params['voc_version']]:
		print ('{:>5s}'.format(cls[:4]), end='', flush=True)
	print ('{:>5s}'.format('mAP'), end='', flush=True)
	print (flush=True)

def PrintAveragePrecision(params, summary):
	for cls in _VOC_CLASSES[params['voc_version']]:
		print ('{:5.1f}'.format(summary['average_precision_summary'][cls] * 100),
			end='', flush=True)
	print ('{:5.1f}'.format(summary['meanAP'] * 100), end='', flush=True)
	print ()

def PrettyPrintSummaryJsons(params, summary_jsons):
	PrintAveragePrecisionHeader(params)
	for print_item in params['print_items']:
		print ('\t[{}]'.format(print_item))
		for result_dir, summary_json in summary_jsons.items():
			print ('\t{}'.format(result_dir))
			PrintAveragePrecision(params, summary_json[print_item])

def _GetArguments():
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--interactive", action="store_true", default=False,
		help="Run the script in an interactive mode")
	parser.add_argument("--voc_version", default="VOC2007",
		help="Target VOC dataset version")
	parser.add_argument("--result_dirs", nargs='+', required=True,
		help="List of target result directories")
	parser.add_argument("--print_items", nargs='+',
		default=['test_summary', 'best_summary'],
		help="List of items to be printed out")
	args = parser.parse_args()
	params = vars(args)
	print (json.dumps(params, indent=2))
	return params	

def main(params):
	summary_jsons = LoadSummaryJsons(params)
	PrettyPrintSummaryJsons(params, summary_jsons)

if __name__ == "__main__":
	params = _GetArguments()
	if not params['interactive']:
		main(params)
