package ai.sklearn4j.core.packaging.loaders;

import ai.sklearn4j.core.packaging.loaders.classifiers.naive_bayes.GaussianNaiveBayesContentLoader;

import java.util.HashMap;
import java.util.Map;

public class ScikitLearnContentLoader {
    private static Map<String, IScikitLearnContentLoader> registeredLoaders = null;

    static {
        registeredLoaders = new HashMap<>();

        addLoader(new GaussianNaiveBayesContentLoader());
    }

    private static void addLoader(IScikitLearnContentLoader loader) {
        registeredLoaders.put(loader.getTypeName(), loader);
    }

    public static IScikitLearnContentLoader loaderForType(String type) {
        if (registeredLoaders.containsKey(type)) {
            return registeredLoaders.get(type);
        }

        throw new RuntimeException(String.format("The loader type '%s' is not supported.", type));
    }
}
