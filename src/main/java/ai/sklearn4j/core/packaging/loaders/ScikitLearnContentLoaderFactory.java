package ai.sklearn4j.core.packaging.loaders;

import ai.sklearn4j.core.ScikitLearnCoreException;
import ai.sklearn4j.core.packaging.loaders.classifiers.naive_bayes.*;

import java.util.HashMap;
import java.util.Map;

/**
 * A factory class that instantiate a loader of type IScikitLearnContentLoader for a given
 * serializer name.
 */
public class ScikitLearnContentLoaderFactory {
    /**
     * A map of registered loaders.
     */
    private static Map<String, IScikitLearnContentLoader> registeredLoaders = null;

    /**
     * Static initialization of the class and registering the available loaders.
     */
    static {
        registeredLoaders = new HashMap<>();

        registerLoader(new GaussianNaiveBayesContentLoader());
        registerLoader(new BernoulliNaiveBayesContentLoader());
        registerLoader(new CategoricalNaiveBayesContentLoader());
        registerLoader(new ComplementNaiveBayesContentLoader());
        registerLoader(new MultinomialNaiveBayesContentLoader());
    }

    /**
     * Registers a new loader in the static map.
     *
     * @param loader Instance of the loader to be registered.
     */
    private static void registerLoader(IScikitLearnContentLoader loader) {
        registeredLoaders.put(loader.getTypeName(), loader);
    }

    /**
     * Returns a loader based on its name for deserialization of contents.
     *
     * @param type Name/ID of the loader.
     * @return An IScikitLearnContentLoader associated with the requested loader name.
     */
    public static IScikitLearnContentLoader loaderForType(String type) {
        if (registeredLoaders.containsKey(type)) {
            return registeredLoaders.get(type);
        }

        throw new ScikitLearnCoreException(String.format("The loader type '%s' is not supported.", type));
    }
}
